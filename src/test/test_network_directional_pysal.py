import unittest
import sys
sys.path.append('../')

from src import network_directional as direc
import numpy as np
from src import network as pynet
from src.netvoronoi_cluster import read_events, snap_and_count

class Rose_Tester(unittest.TestCase):
    def setUp(self):
        self.network = pynet.read_network('../../data/pysal/streets.shp')
        events_t0 = read_events('../../data/pysal/crimes.shp')
        events_t1 = read_events('../../data/pysal/crimes.shp')
        [self.eventIdx_t0, self.eventIdx_t1] = snap_and_count([events_t0,events_t1],self.network)

    def test_net_rose(self):
        np.random.seed(100)
        results = direc.net_rose(self.network, self.eventIdx_t0, self.eventIdx_t1, 'Node-based', k=4, permutations=99)
        print(results)

suite = unittest.TestSuite()
test_classes = [Rose_Tester]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)