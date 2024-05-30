import copy
from itertools import product
import random


RESET_PENALTY = 2
DEPOSIT_SCORE = 4
MARINE_COLLISION_PENALTY = 1


class OptimalPirateAgent:
    def __init__(self, initial):
        self.marine_ships = initial['marine_ships']
        self.treasures = initial['treasures']
        self.map, state = create_state(initial)
        self.turns = initial["turns to go"]
        first_ship = list(state[1].keys())[0]
        self.num_of_pirates = len(state[1].keys())
        self.pirate_names = list(state[1].keys())
        self.treasure_names = list(state[2].keys())
        self.marine_names = list(state[3].keys())
        self.base_location = state[1][first_ship]['location']
        self.row_len = len(self.map)
        self.col_len = len(self.map[0]) if self.row_len > 0 else 0
        self.vi_state = self.create_initial_state(initial)
        self.vi_table, self.policy_table = self.value_iteration(state, self.map)


    def reward(self, state):
        pirate_ships = state[0]
        marine_ships = state[2]
        reward = 0
        x, y = pirate_ships['pirate_ship']['location']
        marine_ships_current_locations = []
        for marine, marine_ship in enumerate(marine_ships.keys()):
            marine_ship_og_name = self.marine_names[marine]
            marine_ships_current_locations.append(self.marine_ships[marine_ship_og_name]['path'][marine_ships[marine_ship]['index']])
        if (x, y) in marine_ships_current_locations:
            reward -= MARINE_COLLISION_PENALTY
        return reward

    def build_loc_map(self, state):
        location_map = [[{} for _ in range(self.col_len)] for _ in range(self.row_len)]
        for x in range(self.row_len):
            for y in range(self.col_len):
                location_map[x][y] = {'pirate_ships': [],
                                      "marine_ships": [],
                                      'treasures': []}
        treasures = state[1]
        marine_ships = state[2]
        pirate_ships = state[0]
        x, y = pirate_ships['pirate_ship']['location']
        location_map[x][y]['pirate_ships'].append('pirate_ship')
        for marine_name, m_ship in enumerate(marine_ships.keys()):
            index = marine_ships[m_ship]['index']
            x, y = self.marine_ships[self.marine_names[marine_name]]['path'][index]
            location_map[x][y]['marine_ships'].append(m_ship)
        for treasure in treasures.keys():
            x, y = treasures[treasure]['location']
            location_map[x][y]['treasures'].append(treasure)
        return location_map

    def possible_actions(self, state):
        location_map = self.build_loc_map(state)
        map_rows = len(location_map)
        map_columns = len(location_map[0])
        pirate_ships = state[0]
        # sail
        pirate_ship = 'pirate_ship'
        possible_actions = []
        i, j = pirate_ships[pirate_ship]['location']
        if j != 0:
            if self.map[i][j - 1] != 'I':
                possible_actions.append(("sail", pirate_ship, (i, j - 1)))
            else:
                if len(location_map[i][j - 1]['treasures']) > 0 and pirate_ships[pirate_ship]['capacity'] > 0:
                    for t in location_map[i][j - 1]['treasures']:
                        possible_actions.append(("collect", pirate_ship, t))
                        break
        if j != map_columns - 1:
            if self.map[i][j + 1] != 'I':
                possible_actions.append(("sail", pirate_ship, (i, j + 1)))
            else:
                if (len(location_map[i][j + 1]['treasures']) > 0 and pirate_ships[pirate_ship]['capacity'] > 0):
                    for t in location_map[i][j + 1]['treasures']:
                        possible_actions.append(("collect", pirate_ship, t))
                        break
        if i != 0:
            if self.map[i - 1][j] != 'I':
                possible_actions.append(("sail", pirate_ship, (i - 1, j)))
            else:
                if (len(location_map[i - 1][j]['treasures']) > 0 and pirate_ships[pirate_ship]['capacity'] > 0):
                    for t in location_map[i - 1][j]['treasures']:
                        possible_actions.append(("collect", pirate_ship, t))
                        break
        if i != map_rows - 1:
            if self.map[i + 1][j] != 'I':
                possible_actions.append(("sail", pirate_ship, (i + 1, j)))

            else:
                if len(location_map[i + 1][j]['treasures']) > 0 and pirate_ships[pirate_ship]['capacity'] > 0:
                    for t in location_map[i + 1][j]['treasures']:
                        possible_actions.append(("collect", pirate_ship, t))
                        break
        if self.map[i][j] == 'B' and pirate_ships[pirate_ship]['capacity'] < 2:
            possible_actions.append(("deposit", pirate_ship))
        possible_actions.append(("wait", pirate_ship))
        possible_actions.append('reset')
        possible_actions.append('terminate')
        return possible_actions

    def create_vi_state(self, i, j, capacity, m_combination, t_combination):
        pirate_ships = {}
        pirate_ships['pirate_ship'] = {"location": (i, j), "capacity": capacity}
        treasures = {}
        for i, t in enumerate(t_combination):
            treasures[f'treasure_{i}'] = {"location": t}
        marine_ships = {}
        for i, m_i in enumerate(m_combination):
            marine_ships[f'marine_{i}'] = {"index": m_i}
        return (pirate_ships, treasures, marine_ships)

    def hashable_vi(self, state):
        hash_ships = tuple((ship, dict['location'], dict['capacity']) for ship, dict in state[0].items())
        hash_treasures = tuple((k, dict['location']) for k, dict in state[1].items())
        hash_marines = tuple((k, dict['index']) for k, dict in state[2].items())
        return hash_ships + hash_treasures + hash_marines


    def state_probs(self, treasure_dict, marine_dict):
        all_treasure_keys = []
        all_treasure_probs = []
        for treasure in treasure_dict.values():
            keys = list(treasure.keys())
            probs = list(treasure.values())
            all_treasure_keys.append(keys)
            all_treasure_probs.append(probs)
        all_marine_keys = []
        all_marine_probs = []
        for marine in marine_dict.values():
            keys = list(marine.keys())
            probs = list(marine.values())
            all_marine_keys.append(keys)
            all_marine_probs.append(probs)
        products = list(product(*all_treasure_keys, *all_marine_keys))
        probabilities = []
        for combination in products:
            prob = 1.0
            for i, loc in enumerate(combination):
                if i < len(all_treasure_keys):
                    prob *= treasure_dict[list(treasure_dict.keys())[i]][loc]
                else:
                    prob *= marine_dict[list(marine_dict.keys())[i - len(all_treasure_keys)]][loc]
            probabilities.append(prob)
        products_probs = list(zip(products, probabilities))
        return products_probs

    def changes(self, comb, treasure_dict, marine_dict, state):
        new_state = copy.deepcopy(state)
        cur_marine_locs = []
        for i, t in enumerate(treasure_dict.keys()):
            new_state[1][t]['location'] = comb[0][i]
        for j, m in enumerate(marine_dict.keys()):
            new_index = comb[0][j + i + 1]
            new_state[2][m]['index'] = new_index
            cur_marine_locs.append(self.marine_ships[self.marine_names[j]]['path'][new_index])
        return new_state, cur_marine_locs

    def create_state_from_input(self, input):
        first_pirate_ship = self.pirate_names[0]
        loc = input['pirate_ships'][first_pirate_ship]['location']
        pirate_ships = {}
        pirate_ships['pirate_ship'] = {"location": loc, "capacity": input['pirate_ships'][first_pirate_ship]['capacity']}
        treasures = {}
        for i, t in enumerate(input['treasures'].values()):
            treasures[f'treasure_{i}'] = {"location": t['location']}
        marine_ships = {}
        for i, marine in enumerate(input['marine_ships'].values()):
            marine_ships[f'marine_{i}'] = {"index": marine['index']}
        return (pirate_ships, treasures, marine_ships)

    def create_initial_state(self, input):
        i, j = self.base_location
        pirate_ships = {}
        first_pirate_ship = self.pirate_names[0]
        pirate_ships['pirate_ship'] = {"location": (i, j), "capacity": input['pirate_ships'][first_pirate_ship]['capacity']}
        treasures = {}
        for i, t in enumerate(input['treasures'].values()):
            treasures[f'treasure_{i}'] = {"location": t['location']}
        marine_ships = {}
        for i, marine in enumerate(input['marine_ships'].values()):
            marine_ships[f'marine_{i}'] = {"index": marine['index']}
        return (pirate_ships, treasures, marine_ships)

    def transition(self, state, action):
        pirate_ships = state[0]
        treasures = state[1]
        marine_ships = state[2]
        probabilities_dict = {}
        action_name = action[0]
        treasure_dict = {}
        marine_dict = {}

        for treasure_name, treasure in enumerate(treasures.keys()):
            treasure_dict[treasure] = {}
            original_name = self.treasure_names[treasure_name]
            transition_prob = self.treasures[original_name]['prob_change_location']
            for location in self.treasures[original_name]['possible_locations']:
                if location == treasures[treasure]['location']:
                    treasure_dict[treasure][location] = (1-transition_prob) + transition_prob/len(self.treasures[original_name]['possible_locations'])
                else:
                    treasure_dict[treasure][location] = transition_prob/len(self.treasures[original_name]['possible_locations'])
        for marine_name, marine in enumerate(marine_ships.keys()):
            original_name = self.marine_names[marine_name]
            if len(self.marine_ships[original_name]['path']) == 1:
                marine_dict[marine] = {0: 1}
            else:
                if marine_ships[marine]['index'] == 0:
                    marine_dict[marine] = {0: 0.5, 1: 0.5}
                elif marine_ships[marine]['index'] == len(self.marine_ships[original_name]['path']) - 1:
                    marine_dict[marine] = {len(self.marine_ships[original_name]['path']) - 1: 0.5, len(self.marine_ships[original_name]['path']) - 2: 0.5}
                else:
                    marine_dict[marine] = {marine_ships[marine]['index'] - 1: 0.3333, marine_ships[marine]['index']: 0.3333, marine_ships[marine]['index'] + 1: 0.3333}
        current_loc = pirate_ships['pirate_ship']['location']
        env_combinations = self.state_probs(treasure_dict, marine_dict)
        if action_name == 'wait':
            for comb in env_combinations:
                new_state, current_marine_locs = self.changes(comb, treasure_dict, marine_dict, state)
                if current_loc in current_marine_locs:
                    new_state[0]['pirate_ship']['capacity'] = 2
                probabilities_dict[self.hashable_vi(new_state)] = comb[1]
        if action_name == 'sail':
            new_loc = action[2]
            for comb in env_combinations:
                new_state, current_marine_locs = self.changes(comb, treasure_dict, marine_dict, state)
                new_state[0]['pirate_ship']['location'] = new_loc
                if new_loc in current_marine_locs:
                    new_state[0]['pirate_ship']['capacity'] = 2
                probabilities_dict[self.hashable_vi(new_state)] = comb[1]
        if action_name == 'collect':
            for comb in env_combinations:
                new_state, current_marine_locs = self.changes(comb, treasure_dict, marine_dict, state)
                new_state[0]['pirate_ship']['capacity'] -= 1
                if current_loc in current_marine_locs:
                    new_state[0]['pirate_ship']['capacity'] = 2
                probabilities_dict[self.hashable_vi(new_state)] = comb[1]
        if action_name == 'deposit':
            for comb in env_combinations:
                new_state, _ = self.changes(comb, treasure_dict, marine_dict, state)
                new_state[0]['pirate_ship']['capacity'] = 2
                probabilities_dict[self.hashable_vi(new_state)] = comb[1]
        if action_name == 'reset':
            initial_state = self.vi_state
            probabilities_dict[self.hashable_vi(initial_state)] = 1
        if action_name == 'terminate':
            probabilities_dict = {}
        return probabilities_dict


    def value_iteration(self, initial, map):
        treasures = initial[2]
        marine_ships = initial[3]
        marine_indices = [list(range(len(ship['path']))) for ship in marine_ships.values()]
        treasure_locations = [treasure['possible_locations'] for treasure in treasures.values()]
        combinations_of_tlocs = list(product(*treasure_locations))
        combinations_of_paths = list(product(*marine_indices))
        V = [{} for _ in range(self.turns+1)]
        pi = [{} for _ in range(self.turns+1)]
        possible_states = []
        for i in range(len(map)):
            for j in range(len(map[0])):
                if map[i][j] == 'I':
                    continue
                for capacity in range(3):
                    for m_combination in combinations_of_paths:
                        for t_combination in combinations_of_tlocs:
                            current_state = self.create_vi_state(i, j, capacity, m_combination, t_combination)
                            possible_states.append(current_state)
                            hash_state = self.hashable_vi(current_state)
                            V[0][hash_state] = self.reward(current_state)
        for t in range(1, self.turns+1):
            for s in possible_states:
                hash_state = self.hashable_vi(s)
                V[t][hash_state] = float('-inf')
                for action in self.possible_actions(s):
                    value = 0
                    if action[0] == 'deposit':
                        value += DEPOSIT_SCORE * (2 - s[0]['pirate_ship']['capacity'])
                    if action[0] == 'reset':
                        value -= RESET_PENALTY
                    prob_dict = self.transition(s, action)
                    for s_prime in prob_dict.keys():
                        value += prob_dict[s_prime] * V[t-1][s_prime]
                    if value > V[t][hash_state]:
                        pi[t][hash_state] = action
                        V[t][hash_state] = value
                V[t][hash_state] += self.reward(s)
        return V, pi

    def act(self, s):
        state = self.create_state_from_input(s)
        current_turn = s['turns to go']
        hash_state = self.hashable_vi(state)
        best_action = self.policy_table[current_turn][hash_state]
        action_name = best_action[0]
        best_action_extended = []
        for p in self.pirate_names:
            if action_name == 'sail':
                best_true_action = (best_action[0], p, best_action[2])
                best_action_extended.append(best_true_action)
            elif action_name == 'collect':
                treasure_index = int(best_action[2].split('_')[-1])
                best_true_action = (best_action[0], p, self.treasure_names[treasure_index])
                best_action_extended.append(best_true_action)
            elif action_name == 'deposit' or action_name == 'wait':
                best_true_action = (best_action[0], p)
                best_action_extended.append(best_true_action)
            elif action_name == 'reset' or action_name == 'terminate':
                return best_action
            else:
                best_action_extended.append(best_action)
        return tuple(best_action_extended)


class PirateAgent:
    def __init__(self, initial):

        self.map, state = create_state(initial)
        self.marine_ships = initial['marine_ships']
        self.treasures = initial['treasures']
        new_turns_num = 10
        self.map, state = create_state(initial)
        self.turns = initial["turns to go"]
        self.num_of_pirates = len(state[1].keys())
        self.pirate_names = list(state[1].keys())
        self.treasure_names = list(state[2].keys())
        self.marine_names = list(state[3].keys())
        first_ship = list(state[1].keys())[0]
        self.base_location = state[1][first_ship]['location']
        self.marine_locations = set(cell for marine_info in self.marine_ships.values() for cell in marine_info['path'])
        self.rows = len(self.map)
        self.cols = len(self.map[0]) if self.rows > 0 else 0
        self.new_marines = self.create_new_marines(state[3])
        best_t_num = 2
        if self.rows > 10 and self.cols > 10:
            best_t_num = 1
        self.best_treasure_names = self.find_best_treasures(self.treasures, best_t_num)
        self.best_treasures = {t: state[2][t] for t in self.best_treasure_names}

        new_state = (copy.deepcopy(state[0]), copy.deepcopy(state[1]), self.best_treasures, self.new_marines)

        initial['marine_ships'] = self.new_marines
        initial['turns to go'] = new_turns_num
        initial['treasures'] = self.best_treasures
        self.initial_vi_state = self.create_initial_state(initial)
        self.value_iteration_table, self.policy_table = self.value_iteration(new_state, self.map)

    def create_new_marines(self, marine_ships):
        new_marines = {}
        for i, m in enumerate(marine_ships.keys()):
            new_marines[m] = {"index": marine_ships[m]['index']}
        new_marine_ships = {}
        for marine, details in marine_ships.items():
            for cell in details["path"]:
                new_marine_ships[f"{marine}_{cell}"] = {"index": details["index"], "path": [cell]}
        return new_marine_ships

    def reward(self, state):
        pirate_ships = state[0]
        reward = 0
        i, j = pirate_ships['pirate_ship']['location']
        if (i, j) in self.marine_locations:
            reward -= MARINE_COLLISION_PENALTY
        return reward

    def build_loc_map(self, state):
        location_map = [[{} for _ in range(self.cols)] for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(self.cols):
                location_map[i][j] = {'pirate_ships': [],
                                      "marine_ships": [],
                                      'treasures': []}
        pirate_ships = state[0]
        treasures = state[1]
        i, j = pirate_ships['pirate_ship']['location']
        location_map[i][j]['pirate_ships'].append('pirate_ship')
        for m_loc in self.marine_locations:
            i, j = m_loc
            location_map[i][j]['marine_ships'].append('marine')
        for treasure in treasures.keys():
            i, j = treasures[treasure]['location']
            location_map[i][j]['treasures'].append(treasure)
        return location_map

    def possible_actions(self, state):
        location_map = self.build_loc_map(state)
        map_rows = len(location_map)
        map_columns = len(location_map[0])
        pirate_ships = state[0]
        pirate_ship = 'pirate_ship'
        possible_actions = []
        i, j = pirate_ships[pirate_ship]['location']
        if j != 0:
            if self.map[i][j - 1] != 'I':
                possible_actions.append(("sail", pirate_ship, (i, j - 1)))
            else:
                if len(location_map[i][j - 1]['treasures']) > 0 and pirate_ships[pirate_ship]['capacity'] > 0:
                    for t in location_map[i][j - 1]['treasures']:
                        possible_actions.append(("collect", pirate_ship, t))
                        break
        if j != map_columns - 1:
            if self.map[i][j + 1] != 'I':
                possible_actions.append(("sail", pirate_ship, (i, j + 1)))
            else:
                if (len(location_map[i][j + 1]['treasures']) > 0 and pirate_ships[pirate_ship]['capacity'] > 0):
                    for t in location_map[i][j + 1]['treasures']:
                        possible_actions.append(("collect", pirate_ship, t))
                        break
        if i != 0:
            if self.map[i - 1][j] != 'I':
                possible_actions.append(("sail", pirate_ship, (i - 1, j)))
            else:
                if (len(location_map[i - 1][j]['treasures']) > 0 and pirate_ships[pirate_ship]['capacity'] > 0):
                    for t in location_map[i - 1][j]['treasures']:
                        possible_actions.append(("collect", pirate_ship, t))
                        break
        if i != map_rows - 1:
            if self.map[i + 1][j] != 'I':
                possible_actions.append(("sail", pirate_ship, (i + 1, j)))
            else:
                if len(location_map[i + 1][j]['treasures']) > 0 and pirate_ships[pirate_ship]['capacity'] > 0:
                    for t in location_map[i + 1][j]['treasures']:
                        possible_actions.append(("collect", pirate_ship, t))
                        break
        if self.map[i][j] == 'B' and pirate_ships[pirate_ship]['capacity'] < 2:
            possible_actions.append(("deposit", pirate_ship))
        possible_actions.append(("wait", pirate_ship))
        possible_actions.append('reset')
        possible_actions.append('terminate')
        return possible_actions

    def create_vi_state(self, x, y, capacity, marine_product, treasure_product):
        pirate_ships = {}
        pirate_ships['pirate_ship'] = {"location": (x, y), "capacity": capacity}
        treasures = {}
        for x, t in enumerate(treasure_product):
            treasures[f'treasure_{x}'] = {"location": t}
        marine_ships = {}
        marine_names = list(self.new_marines.keys())
        for x, treasure in enumerate(marine_product):
            marine_ships[marine_names[x]] = {"index": treasure}
        return (pirate_ships, treasures, marine_ships)


    def environment_probs(self, treasure_dict, marine_dict):
        all_treasure_keys = []
        all_treasure_probs = []
        for treasure in treasure_dict.values():
            probs = list(treasure.values())
            keys = list(treasure.keys())
            all_treasure_probs.append(probs)
            all_treasure_keys.append(keys)
        all_marine_keys = []
        all_marine_probs = []
        for marine in marine_dict.values():
            keys = list(marine.keys())
            probs = list(marine.values())
            all_marine_keys.append(keys)
            all_marine_probs.append(probs)
        products = list(product(*all_treasure_keys, *all_marine_keys))
        probabilities = []
        for prod in products:
            prob = 1
            for i, loc in enumerate(prod):
                if i < len(all_treasure_keys):
                    prob *= treasure_dict[list(treasure_dict.keys())[i]][loc]
                else:
                    prob *= marine_dict[list(marine_dict.keys())[i - len(all_treasure_keys)]][loc]
            probabilities.append(prob)
        products_probs = list(zip(products, probabilities))
        return products_probs

    def environment_changes(self, comb, treasure_dict, marine_dict, state):
        new_state = copy.deepcopy(state)
        for i, t in enumerate(treasure_dict.keys()):
            new_state[1][t]['location'] = comb[0][i]
        cur_marine_locs = self.marine_locations
        return new_state, cur_marine_locs

    def create_state_from_input(self, input):
        first_pirate_ship = self.pirate_names[0]
        loc = input['pirate_ships'][first_pirate_ship]['location']
        pirate_ships = {}
        pirate_ships['pirate_ship'] = {"location": loc,
                                       "capacity": input['pirate_ships'][first_pirate_ship]['capacity']}
        treasures = {}
        for i, treasure in enumerate(self.best_treasure_names):
            treasures[f'treasure_{i}'] = {"location": input['treasures'][treasure]['location']}
        marine_ships = {}
        for marine in self.new_marines.keys():
            marine_ships[marine] = {"index": 0}
        return (pirate_ships, treasures, marine_ships)

    def create_initial_state(self, input):
        i, j = self.base_location
        pirate_ships = {}
        first_pirate_ship = self.pirate_names[0]
        pirate_ships['pirate_ship'] = {"location": (i, j),
                                       "capacity": input['pirate_ships'][first_pirate_ship]['capacity']}
        treasures = {}
        for i, t in enumerate(input['treasures'].values()):
            treasures[f'treasure_{i}'] = {"location": t['location']}
        marine_ships = {}
        for marine, marine_info in input['marine_ships'].items():
            marine_ships[marine] = {"index": 0}
        return (pirate_ships, treasures, marine_ships)

    def transition(self, state, action):
        pirate_ships = state[0]
        treasures = state[1]
        marine_ships = state[2]
        probabilities_dict = {}
        action_name = action[0]
        treasure_dict = {}
        marine_dict = {}
        for treasure_name, treasure in enumerate(treasures.keys()):
            treasure_dict[treasure] = {}
            original_name = self.best_treasure_names[treasure_name]
            transition_prob = self.treasures[original_name]['prob_change_location']
            for location in self.treasures[original_name]['possible_locations']:
                if location == treasures[treasure]['location']:
                    treasure_dict[treasure][location] = (1 - transition_prob) + transition_prob / len(
                        self.treasures[original_name]['possible_locations'])
                else:
                    treasure_dict[treasure][location] = transition_prob / len(
                        self.treasures[original_name]['possible_locations'])

        for marine_name, marine in enumerate(self.new_marines.keys()):
            path_len = len(self.new_marines[marine]['path'])
            if path_len == 1:
                marine_dict[marine] = {0: 1}
            else:
                if marine_ships[marine]['index'] == 0:
                    marine_dict[marine] = {0: 0.5, 1: 0.5}
                elif marine_ships[marine]['index'] == path_len - 1:
                    marine_dict[marine] = {path_len - 1: 0.5,
                                           path_len - 2: 0.5}
                else:
                    marine_dict[marine] = {marine_ships[marine]['index'] - 1: 0.3333,
                                           marine_ships[marine]['index']: 0.3333,
                                           marine_ships[marine]['index'] + 1: 0.3333}

        current_loc = pirate_ships['pirate_ship']['location']
        env_combinations = self.environment_probs(treasure_dict, marine_dict)
        if action_name == 'wait':
            for comb in env_combinations:
                new_state, current_marine_locs = self.environment_changes(comb, treasure_dict, marine_dict, state)
                if current_loc in current_marine_locs:
                    new_state[0]['pirate_ship']['capacity'] = 2
                probabilities_dict[self.hash_state(new_state)] = comb[1]
        if action_name == 'sail':
            new_loc = action[2]
            for comb in env_combinations:
                new_state, current_marine_locs = self.environment_changes(comb, treasure_dict, marine_dict, state)
                new_state[0]['pirate_ship']['location'] = new_loc
                if new_loc in current_marine_locs:
                    new_state[0]['pirate_ship']['capacity'] = 2
                probabilities_dict[self.hash_state(new_state)] = comb[1]
        if action_name == 'collect':
            for comb in env_combinations:
                new_state, current_marine_locs = self.environment_changes(comb, treasure_dict, marine_dict, state)
                new_state[0]['pirate_ship']['capacity'] -= 1
                if current_loc in current_marine_locs:
                    new_state[0]['pirate_ship']['capacity'] = 2
                probabilities_dict[self.hash_state(new_state)] = comb[1]
        if action_name == 'deposit':
            for comb in env_combinations:
                new_state, _ = self.environment_changes(comb, treasure_dict, marine_dict, state)
                new_state[0]['pirate_ship']['capacity'] = 2
                probabilities_dict[self.hash_state(new_state)] = comb[1]
        if action_name == 'reset':
            initial_state = self.initial_vi_state
            probabilities_dict[self.hash_state(initial_state)] = 1
        if action_name == 'terminate':
            probabilities_dict = {}
        return probabilities_dict


    def value_iteration(self, initial, map):
        treasures = initial[2]
        marine_ships = initial[3]
        marine_indices = [list(range(len(ship['path']))) for ship in marine_ships.values()]
        treasure_locations = [treasure['possible_locations'] for treasure in treasures.values()]
        combinations_of_tlocs = list(product(*treasure_locations))
        combinations_of_paths = list(product(*marine_indices))
        V = [{} for _ in range(self.turns + 1)]
        pi = [{} for _ in range(self.turns + 1)]
        possible_states = []
        # rows
        for i in range(len(map)):
            # cols
            for j in range(len(map[0])):
                if map[i][j] == 'I':
                    continue
                for capacity in range(3):
                    for m_combination in combinations_of_paths:
                        for t_combination in combinations_of_tlocs:
                            current_state = self.create_vi_state(i, j, capacity, m_combination, t_combination)
                            possible_states.append(current_state)
                            hash_state = self.hash_state(current_state)
                            V[0][hash_state] = self.reward(current_state)

        for t in range(1, self.turns + 1):
            for s in possible_states:
                hash_state = self.hash_state(s)
                V[t][hash_state] = float('-inf')
                for action in self.possible_actions(s):
                    value = 0
                    if action[0] == 'deposit':
                        value += 4 * (2 - s[0]['pirate_ship']['capacity'])
                    if action[0] == 'reset':
                        value -= RESET_PENALTY
                    prob_dict = self.transition(s, action)
                    for s_prime in prob_dict.keys():
                        value += prob_dict[s_prime] * V[t - 1][s_prime]
                    if value > V[t][hash_state]:
                        pi[t][hash_state] = action
                        V[t][hash_state] = value
                V[t][hash_state] += self.reward(s)
        return V, pi

    def act(self, s):
        state = self.create_state_from_input(s)
        current_turn = s['turns to go']
        hash_state = self.hash_state(state)
        best_action = self.policy_table[current_turn][hash_state]
        action_name = best_action[0]
        best_action_extended = []
        for p in self.pirate_names:
            if action_name == 'sail':
                best_true_action = (best_action[0], p, best_action[2])
                best_action_extended.append(best_true_action)
            elif action_name == 'collect':
                treasure_index = int(best_action[2].split('_')[-1])
                best_true_action = (best_action[0], p, self.best_treasure_names[treasure_index])
                best_action_extended.append(best_true_action)
            elif action_name == 'deposit' or action_name == 'wait':
                best_true_action = (best_action[0], p)
                best_action_extended.append(best_true_action)
            elif action_name == 'reset' or action_name == 'terminate':
                return best_action
            else:
                best_action_extended.append(best_action)
        return tuple(best_action_extended)

    def calculate_treasure_probability(self, treasures, t):
        location_probabilities = {}
        for treasure in treasures.values():
            for location in treasure["possible_locations"]:
                if location not in location_probabilities:
                    location_probabilities[location] = 0
                location_probabilities[location] += 1 - treasure["prob_change_location"]
        total_treasures = len(treasures)
        for location, probability in location_probabilities.items():
            location_probabilities[location] = min(1, probability * total_treasures)
        return location_probabilities

    def most_stable(self, treasures):
        sorted_treasures = sorted(treasures.items(), key=lambda x: x[1]['prob_change_location'])
        most_stable_treasures = [t for t, t_info in sorted_treasures[:2]]
        return most_stable_treasures

    def least_possible_locs(self, treasures):
        sorted_treasures = sorted(treasures.items(), key=lambda x: len(x[1]['possible_locations']))
        least_possible_locs = [t for t, t_info in sorted_treasures if
                               len(t_info['possible_locations']) == len(sorted_treasures[0][1]['possible_locations'])]
        return least_possible_locs[:2]

    def are_seas_nearby(self, map, t_info):
        for loc in t_info['possible_locations']:
            i, j = loc
            seas = []
            if i != 0:
                if map[i-1][j] == 'S' and (i-1, j) not in self.marine_locations:
                    seas.append((i-1, j))
            if i != len(map) - 1:
                if map[i+1][j] == 'S' and (i+1, j) not in self.marine_locations:
                    seas.append((i+1, j))
            if j != 0:
                if map[i][j-1] == 'S' and (i, j-1) not in self.marine_locations:
                    seas.append((i, j-1))
            if j != len(map[0]) - 1:
                if map[i][j+1] == 'S' and (i, j+1) not in self.marine_locations:
                    seas.append((i, j+1))
            if not bool(seas):
                return False
        return True

    def dist_from_base(self, loc):
        return abs(loc[0] - self.base_location[0]) + abs(loc[1] - self.base_location[1])


    def closest_treasures(self, treasures):
        sorted_treasures = sorted(treasures.items(), key=lambda x: self.dist_from_base(x[1]['location']))
        min_dist = sorted_treasures[0][1]['location']
        closest_treasures = [t for t, t_info in sorted_treasures if self.dist_from_base(t_info['location']) == min_dist]
        return closest_treasures[:2]

    def find_best_treasures(self, treasures, k):
        new_treasures_set = set(treasures.keys())
        for t, t_info in treasures.items():
            if not self.are_seas_nearby(self.map, t_info):
                new_treasures_set -= {t}
        treasure_counts = {t: 0 for t in treasures.keys()}
        most_stable_treasures = self.most_stable(treasures)
        least_possible_locs = self.least_possible_locs(treasures)
        closest_treasures = self.closest_treasures(treasures)
        for treasure_set in [most_stable_treasures, least_possible_locs, closest_treasures, new_treasures_set]:
            for treasure in treasure_set:
                treasure_counts[treasure] += 1

        top_k_best_treasures = sorted(treasure_counts.keys(), key=lambda x: treasure_counts[x], reverse=True)[:k]
        return top_k_best_treasures

    def hash_state(self, state):
        hash_treasures = tuple((k, dict['location']) for k, dict in state[1].items())
        hash_marines = tuple((k, dict['index']) for k, dict in state[2].items())
        hash_ships = tuple((ship, dict['location'], dict['capacity']) for ship, dict in state[0].items())
        return hash_ships + hash_treasures + hash_marines

    def unhash(self, hashable_state):
        ships = {}
        treasures = {}
        marines = {}
        for item in hashable_state[0]:
            ship_name, location, capacity = item
            ships[ship_name] = {"location": location, "capacity": capacity}
        for item in hashable_state[1]:
            item_name, location = item
            treasures[item_name] = {"location": location}
        for item in hashable_state[2]:
            item_name, index = item
            marines[item_name] = {"index": index}
        return [ships, treasures, marines]


class InfinitePirateAgent:
    def __init__(self, initial, gamma):
        self.initial = initial
        self.gamma = gamma

    def act(self, state):
        raise NotImplemented

    def value(self, state):
        raise NotImplemented


def create_state(state):
    initial = copy.deepcopy(state)
    initial = initial
    map = initial["map"]
    rows = len(map)
    cols = len(map[0]) if rows > 0 else 0
    location_map = [[{} for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            location_map[i][j] = {'pirate_ships': [],
                                       "marine_ships": [],
                                       'treasures': []}
    pirate_ships = initial["pirate_ships"]
    treasures = initial["treasures"]
    marine_ships = initial["marine_ships"]
    for p_ship in pirate_ships.keys():
        i, j = pirate_ships[p_ship]['location']
        location_map[i][j]['pirate_ships'].append(p_ship)
    for m_ship in marine_ships.keys():
        index = marine_ships[m_ship]['index']
        i, j = marine_ships[m_ship]['path'][index]
        location_map[i][j]['marine_ships'].append(m_ship)
    for treasure in treasures.keys():
        i, j = treasures[treasure]['location']
        location_map[i][j]['treasures'].append(treasure)
    return map, (location_map, pirate_ships, treasures, marine_ships)

def hashable(state):
    hash_map = tuple(tuple((key, tuple(value)) for key, value in cell.items()) for row in state[0] for cell in row)
    hash_ships = tuple((ship, tuple(dict['location'])) for ship, dict in state[1].items())
    hash_treasures = tuple((k, dict['location']) for k, dict in state[2].items())
    hash_marines = tuple((k, dict['index'], tuple(dict['path'])) for k, dict in state[3].items())
    return hash_map + hash_ships + hash_treasures + hash_marines

