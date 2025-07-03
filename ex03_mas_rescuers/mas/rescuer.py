import os
import random
from vs.abstract_agent import AbstAgent
import numpy as np
from sklearn.cluster import KMeans
from vs.constants import VS
import heapq
import itertools
import joblib


## Classe que define o Agente Rescuer com um plano fixo
class Rescuer(AbstAgent):
    def __init__(self, env, config_file, nb_of_explorers=1,clusters=[]):
        """ 
        @param env: a reference to an instance of the environment class
        @param config_file: the absolute path to the agent's config file
        @param nb_of_explorers: number of explorer agents to wait for
        @param clusters: list of clusters of victims in the charge of this agent"""

        super().__init__(env, config_file)

        # Specific initialization for the rescuer
        self.nb_of_explorers = nb_of_explorers       # number of explorer agents to wait for start
        self.received_maps = 0                       # counts the number of explorers' maps
        self.walls = set()                           # a set of wall positions
        self.danger_map = {}                 # a dictionary of danger values for each position
        self.victims = {}            # a dictionary of found victims: [vic_id]: {'pos':(x,y), 'vs': [<vs>], 'severity': s}
        self.plan = []               # a list of planned actions in increments of x and y
        self.x = 0                   # the current x position of the rescuer when executing the plan
        self.y = 0                   # the current y position of the rescuer when executing the plan
        self.clusters = clusters     # the clusters of victims this agent should take care of - see the method cluster_victims
        self.victimsOrder = []       # the order of victims to rescue
        
        self.set_state(VS.IDLE)
        
        # Load models
        try:
            # IMPORTANT: Please update these paths to your .joblib files
            self.classifier = joblib.load('ml/best_classifier_dt.joblib')
            self.regressor = joblib.load('ml/best_regressor_dt.joblib')
            print(f"{self.NAME}: Models loaded successfully.")
        except FileNotFoundError:
            self.classifier = None
            self.regressor = None
            print(f"{self.NAME}: WARNING - Models not found. Using random severity.")

    def cluster_victims(self):
        """
        Groups victims into 4 clusters using the K-Means algorithm based on their coordinates.

        @return: A list of lists, where each inner list contains the victim IDs of a cluster.
                 Example: [[vic_id_1, vic_id_5], [vic_id_2], [vic_id_3, vic_id_4], [vic_id_6]]
        """
        
        victim_ids = list(self.victims.keys())
        positions = np.array([self.victims[vic_id]['pos'] for vic_id in victim_ids])

        num_clusters = 4
        
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        kmeans.fit(positions)
        
        labels = kmeans.labels_

        clustered_victims = [[] for _ in range(num_clusters)]

        for i, vic_id in enumerate(victim_ids):
            cluster_index = labels[i]
            clustered_victims[cluster_index].append(vic_id)
        
        print(f"{self.NAME}: Victims clustered successfully into {num_clusters} groups.")

        for i, cluster in enumerate(clustered_victims):
            print(f"{self.NAME}: Cluster {i} contains victims: {cluster}")
            print(f"positions:")
            for vic_id in cluster:
                print(f"{self.victims[vic_id]['pos']}, ")

            filename = f"cluster-{i}.txt"
            with open(filename, "w") as f:
                for vic_id in cluster:
                    pos = self.victims[vic_id]['pos']
                    f.write(f"{vic_id},{pos[0]},{pos[1]},{self.victims[vic_id]['severity_value']},{self.victims[vic_id]['severity']+1}\n")
            filename = f"cluster.txt"
            with open(filename, "w") as f:
                for vic_id in cluster:
                    pos = self.victims[vic_id]['pos']
                    f.write(f"{vic_id},{pos[0]},{pos[1]},{self.victims[vic_id]['severity_value']},{self.victims[vic_id]['severity']+1}\n")
            print(f"Saved cluster {i} to {filename}")
        
        return clustered_victims

    def sync_explorers(self, explorer_walls, explorer_victims, danger_map):
        self.received_maps += 1

        self.walls.update(explorer_walls)
        self.danger_map.update(danger_map)

        # Merge victims
        for vic_id, vs_tuple, pos in explorer_victims:
            if vic_id not in self.victims:
                self.victims[vic_id] = {'pos': pos, 'vs': list(vs_tuple)}
                print(f"{self.NAME}: Added new victim {vic_id} at {pos}")

        if self.received_maps == self.nb_of_explorers:
            # Master rescuer (self) will handle the clustering and distribution
            self.predict_severity_and_class()
            self.clusters = self.cluster_victims()
            self._create_and_coordinate_rescuers()
            self._handle_own_cluster()

    def _create_and_coordinate_rescuers(self):
        """Creates and coordinates other rescuers, distributing clusters among them."""
        # Create 3 additional rescuers
        other_rescuers = []
        for i in range(3):
            current_folder = os.path.abspath(os.getcwd())
            config_ag_folder = os.path.abspath(os.path.join(current_folder, "ex03_mas_rescuers", "cfg_1"))
            rescuer_file = os.path.join(config_ag_folder, f"rescuer_{i+2}_config.txt")

            new_rescuer = Rescuer(self.get_env(), rescuer_file, nb_of_explorers=self.nb_of_explorers)
            new_rescuer.victims = self.victims.copy()
            new_rescuer.danger_map = self.danger_map.copy()
            new_rescuer.walls = self.walls.copy()
            other_rescuers.append(new_rescuer)
            print(f"{self.NAME}: Created rescuer {new_rescuer.NAME}")

        # Distribute clusters to rescuers
        for i, rescuer in enumerate(other_rescuers):
            if i + 1 < len(self.clusters):  # +1 because master takes first cluster
                cluster = self.clusters[i + 1]
                # Share necessary information with the rescuer
                rescuer.walls = self.walls.copy()
                rescuer.victims = {vic_id: self.victims[vic_id].copy() for vic_id in cluster}
                rescuer.clusters = [cluster]  # Each rescuer gets one cluster
                print(f"{self.NAME}: Assigned cluster {i+1} to {rescuer.NAME}")
                
                # Start the rescuer's planning process
                rescuer._start_planning()

    def _handle_own_cluster(self):
        """Handles the master rescuer's own cluster."""
        if not self.clusters:
            return

        # Master rescuer takes the first cluster
        my_cluster = self.clusters[0]
        self.clusters = [my_cluster]  # Update clusters to only contain our cluster
        
        # Run planning process
        self._start_planning()

    def _start_planning(self):
        """Starts the planning process for this rescuer."""
        print(f"{self.NAME}: Starting planning process...")
        
        # Run genetic algorithm for sequencing
        self.sequencing()
        
        # Generate the detailed action plan
        # self._generate_plan()
        
        # Set state to active
        self.set_state(VS.ACTIVE)
        print(f"{self.NAME} is now ACTIVE.")

    def predict_severity_and_class(self):
        """
        This method predicts severity class and value for each victim using loaded models.
        If models are not available, it assigns a random severity.
        """
        if self.classifier is None or self.regressor is None:
            # Fallback to random if models are not loaded
            for vic_id in self.victims:
                self.victims[vic_id]['severity'] = random.randint(1, 4)
                self.victims[vic_id]['severity_value'] = self.victims[vic_id]['severity']
            return

        for vic_id, victim_data in self.victims.items():
            # The decision tree models expect 3 features in a specific order:
            # 1. Heart Beat Rate (Pulse Frequency) - index 3
            # 2. Pressure Quality - index 4
            # 3. Respiration Frequency - index 5
            # ['qPA', 'pulse', 'resp_freq']
            vs = victim_data['vs']
            filtered_vs = [vs[3], vs[4], vs[5]]
            vital_signals = np.array(filtered_vs).reshape(1, -1)

            # Predict severity class (1-4)
            severity_class = self.classifier.predict(vital_signals)[0]
            self.victims[vic_id]['severity'] = severity_class
            
            # Predict severity value (1-100)
            severity_value = self.regressor.predict(vital_signals)[0]
            self.victims[vic_id]['severity_value'] = severity_value
        
    def sequencing(self):
        """
        Defines the optimal SUBSET and SEQUENCE of victims to rescue using a Genetic Algorithm.
        - Pre-computes real travel costs using A*.
        - The fitness function simulates the rescue mission, respecting self.TLIM.
        """
        victims = self.clusters[0]
        
        print(f"{self.NAME}: Starting pre-computation of costs for {len(victims)} victims...")
        cost_matrix = self._precompute_costs(victims)
        print(f"{self.NAME}: Pre-computation finished.")

        # --- GA Parameters ---
        POPULATION_SIZE = 100
        NUM_GENERATIONS = 1500
        TOURNAMENT_SIZE = 50
        MUTATION_RATE = 0.1
        CROSSOVER_RATE = 0.8
        
        def calculate_fitness(sequence):
            """
            Calculates fitness by simulating the rescue and summing the severity
            of ONLY the victims that can be rescued within self.TLIM.
            """
            time_spent = 0
            total_severity = 0
            current_id = 'BASE'

            for next_vic_id in sequence:
                cost_to_victim = cost_matrix.get(current_id, {}).get(next_vic_id, {}).get('cost', float('inf'))
                cost_from_victim_to_base = cost_matrix.get(next_vic_id, {}).get('BASE', {}).get('cost', float('inf'))

                # Check if we can visit this victim and still return to base in time
                if time_spent + cost_to_victim + cost_from_victim_to_base + self.COST_FIRST_AID <= self.TLIM:
                    time_spent += cost_to_victim + self.COST_FIRST_AID
                    total_severity += self.victims[next_vic_id].get('severity_value', 0)
                    current_id = next_vic_id
                else:
                    # Not enough time for this victim (and any subsequent ones)
                    break
            
            return total_severity

        def tournament_selection(population, fitnesses):
            """Selects a parent using tournament selection."""
            tournament = random.sample(list(zip(population, fitnesses)), TOURNAMENT_SIZE)
            # Return the best individual from the tournament
            winner = max(tournament, key=lambda item: item[1])
            return winner[0]

        def ordered_crossover(parent1, parent2):
            """Performs ordered crossover (OX)."""
            if random.random() > CROSSOVER_RATE:
                return parent1, parent2
                
            size = len(parent1)
            child1, child2 = [-1]*size, [-1]*size
            
            # Select a random subsequence from parent1
            start, end = sorted(random.sample(range(size), 2))
            
            # Copy the subsequence to child1
            child1[start:end+1] = parent1[start:end+1]
            
            # Fill the rest of child1 with genes from parent2
            p2_genes = [gene for gene in parent2 if gene not in child1]
            
            child1_idx = 0
            for i in range(size):
                if child1[i] == -1:
                    child1[i] = p2_genes.pop(0)
                    
            return child1, child2 # For simplicity, we can just return one child and ignore the second

        def swap_mutation(individual):
            """Performs swap mutation."""
            if random.random() < MUTATION_RATE:
                idx1, idx2 = random.sample(range(len(individual)), 2)
                individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
            return individual

        # --- GA Main Logic ---

        # 1. Initialize Population
        population = []
        for _ in range(POPULATION_SIZE):
            new_sequence = random.sample(victims, len(victims))
            population.append(new_sequence)

        best_sequence_so_far = []
        best_fitness_so_far = -float('inf')

        # 2. Evolution Loop
        for gen in range(NUM_GENERATIONS):
            # Calculate fitness for the whole population
            fitnesses = [calculate_fitness(ind) for ind in population]

            # Track the best individual
            current_best_fitness = max(fitnesses)
            if current_best_fitness > best_fitness_so_far:
                best_fitness_so_far = current_best_fitness
                best_sequence_so_far = population[fitnesses.index(current_best_fitness)]

            # Create the next generation
            new_population = []
            while len(new_population) < POPULATION_SIZE:
                # Selection
                parent1 = tournament_selection(population, fitnesses)
                parent2 = tournament_selection(population, fitnesses)
                
                # Crossover
                child, _ = ordered_crossover(parent1, parent2)
                
                # Mutation
                child = swap_mutation(child)
                
                new_population.append(child)
            
            population = new_population

        self.victimsOrder = best_sequence_so_far
        
        # After the loop, `best_sequence_so_far` holds the best permutation.
        # We must now filter it to get the final, feasible plan.
        final_plan = []
        time_spent = 0
        current_id = 'BASE'
        for vic_id in best_sequence_so_far:
            cost_to_victim = cost_matrix[current_id][vic_id]['cost']
            cost_from_victim_to_base = cost_matrix[vic_id]['BASE']['cost']
            if time_spent + cost_to_victim + cost_from_victim_to_base <= self.TLIM:
                time_spent += cost_to_victim
                final_plan.extend(cost_matrix[current_id][vic_id]['path'])
                current_id = vic_id
            else:
                break
        # add the base to the final plan
        final_plan.extend(cost_matrix[best_sequence_so_far[-1]]['BASE']['path'])
        
        self.plan = final_plan
        

        filename = f"seq{self.NAME}.txt"        
        with open(filename, "w") as f:
            for vic_id in self.victimsOrder:
                pos = self.victims[vic_id]['pos']
                f.write(f"{vic_id},{pos[0]},{pos[1]},{self.victims[vic_id]['severity_value']},{self.victims[vic_id]['severity']}\n")
        print(f"Saved sequence {self.NAME} to {filename}")

    def _calculate_heuristic(self, pos, destination, min_danger):
        """Calculates the Octile Distance heuristic for A*."""
        dx = abs(pos[0] - destination[0])
        dy = abs(pos[1] - destination[1])
        
        min_cost_straight = self.COST_LINE * min_danger
        min_cost_diagonal = self.COST_DIAG * min_danger
        
        return (min(dx, dy) * min_cost_diagonal) + (abs(dx - dy) * min_cost_straight)

    def _reconstruct_path(self, parents, current_pos):
        path = [current_pos]
        while current_pos in parents:
            current_pos = parents[current_pos]
            path.append(current_pos)
        return path[::-1] # Return reversed path (from start to end)

    def a_star_search(self, start_pos, end_pos, danger_map, min_danger):
        """
        Finds the lowest cost path from start to end using A*.
        """
        open_set = [(0, 0, start_pos)]  # (f_cost, g_cost, position)
        parents = {}
        g_costs = {start_pos: 0}

        while open_set:
            _, current_g, current_pos = heapq.heappop(open_set)

            if current_pos == end_pos:
                path = self._reconstruct_path(parents, end_pos)
                return g_costs[end_pos], path

            # Explore 8 neighbors (including diagonals)
            for dx, dy, is_diagonal in [
                (0, 1, False), 
                (0, -1, False), 
                (1, 0, False), 
                (-1, 0, False),
                (1, 1, True), 
                (1, -1, True), 
                (-1, 1, True), 
                (-1, -1, True)
            ]:
                neighbor_pos = (current_pos[0] + dx, current_pos[1] + dy)

                if neighbor_pos in self.walls:
                    continue
                
                # Assume unknown danger is impassable
                danger = danger_map.get(neighbor_pos)
                if danger is None:
                    continue

                move_cost = (self.COST_DIAG if is_diagonal else self.COST_LINE) * danger
                tentative_g_cost = current_g + move_cost

                if tentative_g_cost < g_costs.get(neighbor_pos, float('inf')):
                    parents[neighbor_pos] = current_pos
                    g_costs[neighbor_pos] = tentative_g_cost
                    h_cost = self._calculate_heuristic(neighbor_pos, end_pos, min_danger)
                    f_cost = tentative_g_cost + h_cost
                    heapq.heappush(open_set, (f_cost, tentative_g_cost, neighbor_pos))
        
        # No path found
        return float('inf'), []

    def _precompute_costs(self, victim_ids):
        """
        Pre-computes the travel costs between all victims and the base using A*.
        Returns a matrix (dict of dicts) of costs.
        """
        cost_matrix = {}
        points_of_interest = {'BASE': (0, 0)}
        for vic_id in victim_ids:
            points_of_interest[vic_id] = self.victims[vic_id]['pos']
        
        # Safe minimum danger value for the heuristic
        min_danger = min(filter(None, self.danger_map.values())) if self.danger_map else 0.1

        # Generate all unique pairs of points
        for vic_id1, pos1 in points_of_interest.items():
            for vic_id2, pos2 in points_of_interest.items():
                if vic_id1 == vic_id2:
                    continue

                if vic_id1 not in cost_matrix:
                    cost_matrix[vic_id1] = {}
                if vic_id2 not in cost_matrix:
                    cost_matrix[vic_id2] = {}

                if vic_id1 in cost_matrix[vic_id2]:
                    continue

                cost, path = self.a_star_search(pos1, pos2, self.danger_map, min_danger)
                cost_matrix[vic_id1][vic_id2] = {'cost': cost, 'path': path}
                cost_matrix[vic_id2][vic_id1] = {'cost': cost, 'path': path[::-1]}

        return cost_matrix
    
    def deliberate(self) -> bool:
        # something like this, may change
        if self.plan == []:
           print(f"{self.NAME} has finished the plan")
           return False
        
        pos = self.x, self.y
        while(len(self.plan) > 0 and self.plan[0] == pos):
            self.plan.pop(0)
        
        if self.plan == []:
            print(f"{self.NAME} has finished the plan")
            return False

        x,y = self.plan[0]
        dx, dy = x - self.x, y - self.y

        walked = self.walk(dx, dy)

        if walked == VS.EXECUTED:
            self.x += dx
            self.y += dy

            pos = self.x, self.y
            if len(self.victimsOrder) > 0:
                vic_id = self.victimsOrder[0]
                if pos == self.victims[vic_id]['pos']:
                    self.victimsOrder.pop(0)
                    print(f"{self.NAME} rescued victim {vic_id}")
                    self.first_aid()

        else:
            print(f"{self.NAME} Plan fail - walk error - agent at ({self.x}, {self.x})")

        return True
     
