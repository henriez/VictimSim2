## EXPLORER AGENT
### @Author: Tacla, UTFPR
### It walks randomly in the environment looking for victims.

from enum import Enum
from queue import PriorityQueue
import sys
import os
import random
from abc import ABC, abstractmethod
from vs.abstract_agent import AbstAgent
from vs.constants import VS



class Explorer(AbstAgent):
    class State(Enum):
        EXPLORING = 0
        RETURNING = 1
    
    class AC_INCR(Enum):
        N = (0, -1)
        NE = (1, -1)
        E = (1, 0)
        SE = (1, 1)
        S = (0, 1)
        SW = (-1, 1)
        W = (-1, 0)
        NW = (-1, -1)

    def __init__(self, env, config_file, resc):
        """ Construtor do agente random on-line
        @param env referencia o ambiente
        @config_file: the absolute path to the explorer's config file
        @param resc referencia o rescuer para poder acorda-lo
        """
        super().__init__(env, config_file)
        self.set_state(VS.ACTIVE)

        # Define the order of exploration (clockwise from North)
        # this will change for other explorers in the future
        if self.NAME == 'EXPL_1':
            self.directions_priorities = [self.AC_INCR.N, self.AC_INCR.NE, self.AC_INCR.E, self.AC_INCR.SE, self.AC_INCR.S, self.AC_INCR.SW, self.AC_INCR.W, self.AC_INCR.NW]
        if self.NAME == 'EXPL_2':
            self.directions_priorities = [self.AC_INCR.N, self.AC_INCR.NW, self.AC_INCR.W, self.AC_INCR.SW, self.AC_INCR.S, self.AC_INCR.SE, self.AC_INCR.E, self.AC_INCR.NE]
        if self.NAME == 'EXPL_3':
            self.directions_priorities = [self.AC_INCR.S, self.AC_INCR.SW, self.AC_INCR.W, self.AC_INCR.NW, self.AC_INCR.N, self.AC_INCR.NE, self.AC_INCR.E, self.AC_INCR.SE]
        if self.NAME == 'EXPL_4':
            self.directions_priorities = [self.AC_INCR.S, self.AC_INCR.SE, self.AC_INCR.E, self.AC_INCR.NE, self.AC_INCR.N, self.AC_INCR.NW, self.AC_INCR.W, self.AC_INCR.SW]
        self.path = []
        self.pos = (0,0)
        self.walls = set()
        self.victims = set()
        self.__visited = set()
        self.state = self.State.EXPLORING

        # Heuristics data
        self.heuristics = {}
        self.heuristics[(0,0)] = 0
        self.danger_map = {} # the danger score of each cell
        self.danger_map[self.pos] = VS.OBST_NONE # TODO: intitializing is necessary, analyze if OBST_NONE is a good value
        self.max_cost = self.danger_map[self.pos]

        self.resc = resc           # reference to the rescuer agent   
    
    def deliberate(self) -> bool:
        """ The agent chooses the next action. The simulator calls this
        method at each cycle. Must be implemented in every agent
        Should return False if the agent should stop exploring"""

        print(f"\n{self.NAME} deliberate:")

        if self.NAME == 'EXPL_4':
            print(f"pos: {self.pos}")

        self.__check_walls_and_lim()
        self.__update_heuristics()
        
        if self.pos == (0,0) and self.state == self.State.RETURNING:
            print(f"{self.NAME} Returned successfully, calling rescuer")
            self.resc.sync_explorers(self.walls, self.victims, self.danger_map)
            print(f"{self.NAME}: found {len(self.victims)} victims")
            sev1 = 0
            sev2 = 0
            sev3 = 0
            sev4 = 0
            for vic_id in self.victims:
                if self.get_env().sev_label[vic_id[0]] == 1:
                    sev1 += 1
                elif self.get_env().sev_label[vic_id[0]] == 2:
                    sev2 += 1
                elif self.get_env().sev_label[vic_id[0]] == 3:
                    sev3 += 1
                elif self.get_env().sev_label[vic_id[0]] == 4:
                    sev4 += 1

            print(f"Severidade 1: {sev1}")
            print(f"Severidade 2: {sev2}")
            print(f"Severidade 3: {sev3}")
            print(f"Severidade 4: {sev4}")
            print(f"Total de vítimas: {sev1 + sev2 + sev3 + sev4}")

            return False

        # No more actions, time almost ended
        if self.get_rtime() <= 1.0:
            # time to wake up the rescuer
            # pass the walls and the victims 
            print(f"{self.NAME} No more time to explore... invoking the rescuer")
            # self.resc.go_save_victims([],[])
            print(f"found {len(self.victims)} victims")
            return False

        if self.state == self.State.RETURNING:
            self.__return_to_base()
            return True
        
        if self.get_rtime() <= 1.2*(self.heuristics.get(self.pos) + self.max_cost*max(self.COST_DIAG, self.COST_LINE)): # 10% safety margin
            print(f"{self.NAME} No more time to explore (considering exploration cost)... returning to base")
            vic_set_per_severity = {}
            self.get_env().print_acum_results()
            for v in self.victims:
                print(f"victim id: {v[0]} sev: {self.get_env().sev_label[v[0]]}")
            self.__return_to_base()
            return True
        
        dir = 0
        nextDir = self.directions_priorities[dir]
        nextPos = self.pos[0] + nextDir.value[0], self.pos[1] + nextDir.value[1]

        while (nextPos in self.__visited or self.__is_known_wall(nextDir)) and dir < len(self.directions_priorities):
            nextDir = self.directions_priorities[dir]
            nextPos = self.pos[0] + nextDir.value[0], self.pos[1] + nextDir.value[1]
            dir += 1

        if dir >= len(self.directions_priorities):
            nextDir = self.path.pop()
        else:
            self.path.append(self.__inverse_direction(nextDir)) # add the reverse direction to the path stack

        dx, dy = nextDir.value


        self.__walk_and_update_data(nextDir)
               
        return True

    def __inverse_direction(self, direction):
        if direction == self.AC_INCR.N:
            return self.AC_INCR.S
        elif direction == self.AC_INCR.NE:
            return self.AC_INCR.SW
        elif direction == self.AC_INCR.E:
            return self.AC_INCR.W       
        elif direction == self.AC_INCR.SE:
            return self.AC_INCR.NW
        elif direction == self.AC_INCR.S:
            return self.AC_INCR.N
        elif direction == self.AC_INCR.SW:
            return self.AC_INCR.NE
        elif direction == self.AC_INCR.W:
            return self.AC_INCR.E
        elif direction == self.AC_INCR.NW:
            return self.AC_INCR.SE

    def __check_walls_and_lim(self):
        order = [self.AC_INCR.N, self.AC_INCR.NE, self.AC_INCR.E, self.AC_INCR.SE, self.AC_INCR.S, self.AC_INCR.SW, self.AC_INCR.W, self.AC_INCR.NW]
        walls = self.check_walls_and_lim()
        for i in range(len(order)):
            if walls[i] == VS.WALL or walls[i] == VS.END:
                dir = order[i]
                dx, dy = dir.value
                self.walls.add((self.pos[0]+dx, self.pos[1]+dy))
        
    def __is_known_wall(self, dir):
        dx, dy = dir.value
        return (self.pos[0] + dx, self.pos[1] + dy) in self.walls

    def __update_heuristics(self):
        self.__visited.add(self.pos)

        mnHeuristics = self.heuristics.get(self.pos) if self.pos in self.heuristics else float("inf")
        for d in self.directions_priorities:
            dx, dy = d.value
            px, py = self.pos[0]+dx, self.pos[1]+dy
            
            if (px,py) not in self.heuristics:
                continue
            
            cost = self.heuristics.get((px,py))
            if self.__is_diag_direction(d):
                cost += self.COST_DIAG * self.danger_map[(px,py)]
            else:
                cost += self.COST_LINE * self.danger_map[(px,py)]

            mnHeuristics = min(mnHeuristics, cost)
        
        self.heuristics[self.pos] = mnHeuristics
    
        pq = PriorityQueue()
        pq.put((mnHeuristics, self.pos))

        while not pq.empty():
            cost, pos = pq.get()

            if self.heuristics.get(pos) < cost:
                continue
            
            heur = self.heuristics.get(pos)

            for d in self.directions_priorities:
                nei_pos = (pos[0]+d.value[0], pos[1]+d.value[1])

                if nei_pos not in self.__visited:
                    continue
                    
                nei_cost = heur
                if self.__is_diag_direction(d):
                    nei_cost += self.COST_DIAG * self.danger_map.get(nei_pos)
                else:
                    nei_cost += self.COST_DIAG * self.danger_map.get(nei_pos)

                if nei_cost < self.heuristics.get(nei_pos):
                    self.heuristics[nei_pos] = nei_cost
                    pq.put((nei_cost, nei_pos))
                      
    def __return_to_base(self):
        self.state = self.State.RETURNING
        # TODO: implement return to base based in A*
        # iterating over the direction priorities

        mn_heur_cost = float("inf")
        mn_heur_dir = self.directions_priorities[0]
        for d in self.directions_priorities:
            nei_pos = self.pos[0] + d.value[0], self.pos[1] + d.value[1]
            if not nei_pos in self.__visited:
                continue
            move_cost = self.COST_DIAG if self.__is_diag_direction(d) else self.COST_LINE
            move_cost *= self.danger_map.get(nei_pos)
            if move_cost + self.heuristics.get(nei_pos) < mn_heur_cost:
                mn_heur_cost = move_cost + self.heuristics.get(nei_pos)
                mn_heur_dir = d

        dx, dy = mn_heur_dir.value
        self.heuristics[self.pos] += 1
        self.__walk_and_update_data(mn_heur_dir)

        return True
    
    def __is_diag_direction(self, dir):
        return dir in [self.AC_INCR.NE, self.AC_INCR.SE, self.AC_INCR.SW, self.AC_INCR.NW]

    def __walk_and_update_data(self, dir, read_victim=True):
        prevTime = self.get_rtime()
        dx, dy = dir.value
        result = self.walk(dx, dy)
        spentTime = prevTime - self.get_rtime()

        # updates the cost of the current cell (it can change with time)
        move_cost = self.COST_DIAG if self.__is_diag_direction(dir) else self.COST_LINE

        if result == VS.EXECUTED:
            # TODO: get current pos cost based on spent time
            self.pos = self.pos[0]+dx, self.pos[1]+dy
            self.danger_map[self.pos] = spentTime / move_cost
            self.max_cost = max(self.max_cost, self.danger_map[self.pos])

            if not read_victim:
                return True
            
            vic_id = self.check_for_victim()            # Check if victim was already visited, avoids double signal reading
            if vic_id == VS.NO_VICTIM or vic_id in {v[0] for v in self.victims}:
                return True
            
            # TODO: consider reading cost
            vs = self.read_vital_signals()
            if vs != VS.TIME_EXCEEDED:
                self.victims.add((vic_id, tuple(vs), self.pos))
