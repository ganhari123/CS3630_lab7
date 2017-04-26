import json
import random
import threading
import math


# grid map class
class CozGrid:

    def __init__(self, fname):
        with open(fname) as configfile:
            config = json.loads(configfile.read())
            self.width = config['width']
            self.height = config['height']
            self.scale = config['scale']

            self.occupied = []
            self.markers = []
            self._start = None
            self._goals = []
            self._obstacles = []
            self._visited = set()
            self._newvisited = []
            self._path = []

            # . - empty square
            # O - occupied square
            # U/D/L/R - marker with different orientations up/down/left/right
            for row in range(self.height):
                for col in range(self.width):
                    # empty
                    coord = (col, row)
                    entry = config['layout'][self.height - row - 1][col]
                    if entry == '.':
                        pass
                    # obstacles
                    elif entry == 'O':
                        self.occupied.append((col, row))
                    # marker: U/D/L/R
                    # orientation of markers
                    elif entry == 'U' or entry == 'D' or entry == 'L' or entry == 'R':
                        self.markers.append((col, row, entry))
                    elif entry == 'S':
                        self._start = coord
                    elif entry == 'G':
                        self._goals.append(coord)
                    elif entry == 'X':
                        self._obstacles.append(coord)
                    # error
                    else:
                        raise ValueError('Cannot parse file')
                    # For coordination with visualization
            self.lock = threading.Lock()
            self.updated = threading.Event()
            self.changes = []

    def getNeighbors(self, coord):
        """Get the valid neighbors of a cell and their weights

            Arguments:
            coord -- grid coordinates of grid cell

            Returns list of (coordinate, weight) pairs
        """

        neighbor = []
        x = coord[0]
        y = coord[1]

        for i in range(-1,2,1):
            for j in range(-1,2,1):
                n = (x + i, y + j)
                if n != coord and self.coordInBounds(n) and n not in self._obstacles:
                    neighbor.append((n,math.sqrt(math.pow(i,2) + math.pow(j,2))))

        return neighbor


    def checkPath(self):
        """Checks if the current path is valid, and if so returns its length

            Returns path length as a float if it is valid, -1.0 if it is invalid
        """

        self.lock.acquire()
        pathlen = 0.0
        if len(self._path) > 1:
            current = self._path[0]
            for nextpoint in self._path[1:]:
                neighbors = self.getNeighbors(current)
                in_neighbors = False
                for neighbor in neighbors:
                    if neighbor[0] == nextpoint:
                        pathlen += neighbor[1]
                        in_neighbors = True
                        break
                if not in_neighbors:
                    pathlen = -1.0
                    break
                current = nextpoint

        self.lock.release()
        return pathlen

    def coordInBounds(self, coord):
        """Check if a set of coordinates is in the grid bounds

            Arguments:
            coord -- grid coordinates

            Returns True if coord in bounds, else False
        """

        x = coord[0]
        y = coord[1]
        if x >= 0 and y >= 0 and x < self.width and y < self.height:
            return True


    def addVisited(self, coord):
        """Add a visited cell

            Arguments:
            coord -- grid coordinates of visited cell
        """

        self.lock.acquire()
        self._visited.add(coord)
        self._newvisited.append(coord)
        self.updated.set()
        self.changes.append('visited')
        self.lock.release()


    def getVisited(self):
        """Get the set of visited cells

            Returns: set of coordinates of visited cells
        """

        return self._visited


    def clearVisited(self):
        """Clear the set of visited cells
        """

        self.lock.acquire()
        self._visited = set()
        self.updated.set()
        self.changes.append('allvisited')
        self.lock.release()


    def addObstacle(self, coord):
        """Add an obstacle cell

            Arguments:
            coord -- grid coordinates of obstacle cell
        """

        self.lock.acquire()
        self._obstacles.append(coord)
        self.updated.set()
        self.changes.append('obstacles')
        self.lock.release()


    def addObstacles(self, coords):
        """Add multiple obstacle cells. Useful for marking large objects

            Arguments:
            coords -- list of grid coordinates of obstacle cells
        """

        self.lock.acquire()
        self._obstacles += coords
        self.updated.set()
        self.changes.append('obstacles')
        self.lock.release()


    def clearObstacles(self):
        """Clear list of obstacle cells
        """

        self.lock.acquire()
        self._obstacles = []
        self.updated.set()
        self.changes.append('obstacles')
        self.lock.release()


    def addGoal(self, coord):
        """Add a goal cell

            Arguments:
            coord -- grid coordinates of goal cell
        """

        self.lock.acquire()
        self._goals.append(coord)
        self.updated.set()
        self.changes.append('goals')
        self.lock.release()


    def clearGoals(self):
        """Clear the list of goal cells
        """

        self.lock.acquire()
        self._goals = []
        self.updated.set()
        self.changes.append('goals')
        self.lock.release()


    def getGoals(self):
        """Get the list of goal cells

            Returns list of grid coordinates of goal cells
        """

        return self._goals


    def setStart(self, coord):
        """Set the start cell

            Arguments:
            coord -- grid coordinates of start cell
        """

        self.lock.acquire()
        self._start = coord
        self.updated.set()
        self.changes.append('start')
        self.lock.release()


    def clearStart(self):
        """Clear the start cell
        """

        self.lock.acquire()
        self._start = None
        self.updated.set()
        self.changes.append('start')
        self.lock.release()


    def getStart(self):
        """Get the start cell

            Returns coordinates of start cell
        """

        return self._start


    def setPath(self, path):
        """Set the path

            Arguments:
            path -- list of path coordinates
        """

        self.lock.acquire()
        self._path = path
        self.updated.set()
        self.changes.append('path')
        self.lock.release()


    def getPath(self):
        """Get the path

            Returns list of path coordinates
        """

        return self._path


    def clearPath(self):
        """Clear the path
        """

        self.lock.acquire()
        self._path = []
        self.updated.set()
        self.changes.append('path')
        self.lock.release()


    def is_in(self, x, y):
        """ Determain whether the cell is in the grid map or not
            Argument:
            x, y - X and Y in the cell map
            Return: boolean results
        """
        if x < 0 or y < 0 or x > self.width or y > self.height:
            return False
        return True

    def is_free(self, x, y):
        """ Determain whether the cell is in the *free part* of grid map or not
            Argument:
            x, y - X and Y in the cell map
            Return: boolean results
        """
        if not self.is_in(x, y):
            return False
        yy = int(y) # self.height - int(y) - 1
        xx = int(x)
        return (xx, yy) not in self.occupied

    def random_place(self):
        """ Return a random place in the map
            Argument: None
            Return: x, y - X and Y in the cell map
        """
        x = random.uniform(0, self.width)
        y = random.uniform(0, self.height)
        return x, y

    def random_free_place(self):
        """ Return a random place in the map which is free from obstacles
            Argument: None
            Return: x, y - X and Y in the cell map
        """
        while True:
            x, y = self.random_place()
            if self.is_free(x, y):
                return x, y


# parse marker position and orientation
# input: grid position and orientation char from JSON file
# output: actual marker position (marker origin) and marker orientation
def parse_marker_info(col, row, heading_char):
    if heading_char == 'U':
        c = col + 0.5
        r = row
        heading = 90
    elif heading_char == 'D':
        c = col + 0.5
        r = row + 1
        heading = 270
    elif heading_char == 'L':
        c = col + 1
        r = row + 0.5
        heading = 180
    elif heading_char == 'R':
        c = col
        r = row + 0.5
        heading = 0
    return c, r, heading
