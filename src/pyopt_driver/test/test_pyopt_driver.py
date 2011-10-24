
import unittest

# pylint: disable-msg=E0611,F0401
import numpy
import math
from nose import SkipTest

from pyopt_driver.pyopt_driver import pyOptDriver

from openmdao.util.testutil import assert_rel_error
from openmdao.main.api import Assembly, set_as_top, Component
from openmdao.lib.datatypes.api import Float
from openmdao.lib.differentiators.finite_difference import FiniteDifference
from openmdao.examples.simple.paraboloid import Paraboloid
from openmdao.examples.simple.paraboloid_derivative import ParaboloidDerivative

class OptimizationConstrained(Assembly):
    """Constrained optimization of the Paraboloid with CONMIN."""
    
    def __init__(self):
        """ Creates a new Assembly containing a Paraboloid and an optimizer"""
        
        # pylint: disable-msg=E1101
        
        super(OptimizationConstrained, self).__init__()

        # Create Paraboloid component instances
        self.add('paraboloid', Paraboloid())

        # Create CONMIN Optimizer instance
        self.add('driver', pyOptDriver())
        
        # Driver process definition
        self.driver.workflow.add('paraboloid')
        
        # CONMIN Objective 
        self.driver.add_objective('paraboloid.f_xy')
        
        # CONMIN Design Variables 
        self.driver.add_parameter('paraboloid.x', low=-50., high=50.)
        self.driver.add_parameter('paraboloid.y', low=-50., high=50.)
        
        # CONMIN Constraints
        self.driver.add_constraint('paraboloid.x-paraboloid.y >= 15.0')
        
        self.driver.print_results = False
        
class OptimizationConstrainedDerivatives(Assembly):
    """Constrained optimization of the Paraboloid with CONMIN."""
    
    def __init__(self):
        """ Creates a new Assembly containing a Paraboloid and an optimizer"""
        
        # pylint: disable-msg=E1101
        
        super(OptimizationConstrainedDerivatives, self).__init__()

        # Create Paraboloid component instances
        self.add('paraboloid', ParaboloidDerivative())

        # Create CONMIN Optimizer instance
        self.add('driver', pyOptDriver())
        
        # Driver process definition
        self.driver.workflow.add('paraboloid')
        
        # CONMIN Objective 
        self.driver.add_objective('paraboloid.f_xy')
        
        # CONMIN Design Variables 
        self.driver.add_parameter('paraboloid.x', low=-50., high=50.)
        self.driver.add_parameter('paraboloid.y', low=-50., high=50.)
        
        # CONMIN Constraints
        self.driver.add_constraint('paraboloid.x-paraboloid.y >= 15.0')
        
        self.driver.print_results = False
        
        self.driver.differentiator = FiniteDifference()

        
class MultiFunction(Component):
    #Finds the minimum f(1) = x[1]
    #              and f(2) = (1+x[2])/x[1]
        
    # set up interface to the framework  
    # pylint: disable-msg=E1101
    x1 = Float(1.0, iotype='in', desc='The variable x1')
    x2 = Float(1.0, iotype='in', desc='The variable x2')

    f1_x = Float(iotype='out', desc='f1(x1,x2)')
    f2_x = Float(iotype='out', desc='f2(x1,x2)')

    g1_x = Float(iotype='out', desc='g1(x1,x2)')
    g2_x = Float(iotype='out', desc='g2(x1,x2)')
        
    def execute(self):
        
        x1 = self.x1
        x2 = self.x2
        
        self.f1_x = x1
        self.f2_x = (1+x2)/x1
 
        self.g1_x =  x2+9.0*x1
        self.g2_x = -x2+9.0*x1
        
class MultiObjectiveOptimization(Assembly):
    """Multi Objective optimization of the  with NSGA2."""

    def __init__(self):
        """ Creates a new Assembly containing a MultiFunction and an optimizer"""

        # pylint: disable-msg=E1101

        super(MultiObjectiveOptimization, self).__init__()

        # Create MultiFunction component instances
        self.add('multifunction', MultiFunction())

        # Create NSGA2 Optimizer instance
        self.add('driver', pyOptDriver())

        # Driver process definition
        self.driver.workflow.add('multifunction')

        self.driver.print_results = False

        # NSGA2 Objective
        self.driver.add_objective('multifunction.f1_x')
        self.driver.add_objective('multifunction.f2_x')

        # NSGA2 Design Variable
        self.driver.add_parameter('multifunction.x1', low= 0.1, high=1.0)
        self.driver.add_parameter('multifunction.x2', low= 0.0, high=5.0)

        # NSGA2 Constraints
        self.driver.add_constraint('multifunction.g1_x  >= 6.0')
        self.driver.add_constraint('multifunction.g2_x  >= 1.0')
        
        
class pyOptDriverTestCase(unittest.TestCase):

    def setUp(self):
        pass
        
    def tearDown(self):
        self.top = None
        
    def test_basic_CONMIN(self):
        
        self.top = OptimizationConstrained()
        set_as_top(self.top)
        
        try:
            self.top.driver.optimizer = 'CONMIN'
        except ValueError:
            raise SkipTest("CONMIN not present on this system")
            
        self.top.driver.title='Little Test'
        optdict = {}
        self.top.driver.options = optdict

        self.top.run()
        
        assert_rel_error(self, self.top.paraboloid.x, 7.175775, 0.01)
        assert_rel_error(self, self.top.paraboloid.y, -7.824225, 0.01)

    def test_basic_CONMIN_derivatives(self):
        
        self.top = OptimizationConstrainedDerivatives()
        set_as_top(self.top)
        
        try:
            self.top.driver.optimizer = 'CONMIN'
        except ValueError:
            raise SkipTest("CONMIN not present on this system")
        
        self.top.driver.title='Little Test with Gradient'
        optdict = {}
        self.top.driver.options = optdict

        self.top.run()
        
        assert_rel_error(self, self.top.paraboloid.x, 7.175775, 0.01)
        assert_rel_error(self, self.top.paraboloid.y, -7.824225, 0.01)
        
    def test_GA_multi_obj_multi_con(self):
        # Note, just verifying that things work functionally, rather than run this
        # for many generations.
        
        self.top = MultiObjectiveOptimization()
        set_as_top(self.top)

        try:
            self.top.driver.optimizer = 'NSGA2'
        except ValueError:
            raise SkipTest("NSGA2 not present on this system")
        
        # PyOpt Flags
        self.top.driver.title='Two-Objective Fitness MultiFunction Test'
        optdict = {}
        optdict['PopSize'] = 100     #   a multiple of 4
        optdict['maxGen'] = 5  
        optdict['pCross_real'] = 0.6 #prob of crossover of design variables in range (0.6-1.0)
        optdict['pMut_real'] = 0.5   #prob of mutation of (1/design varaibles)
        optdict['eta_c'] = 10.0      #distribution index for crossover in range (5 - 20)
        optdict['eta_m'] = 50.0      #distribution index for mutation in range (5 - 50)
        optdict['pCross_bin'] = 1.0  #prob of crossover of binary variable in range(0.6 - 1.0)
        optdict['pMut_real'] = 1.0   #prob of mutation of binary variables in (1/nbits)
        optdict['PrintOut'] = 0      #flag to turn on output to files (0-None, 1-Subset,2-All)
        optdict['seed'] = 0.0        #random seed number (0-autoseed based on time clock)

        self.top.driver.options = optdict
        
        self.top.run()
        
        
        
if __name__ == "__main__":
    unittest.main()
    
