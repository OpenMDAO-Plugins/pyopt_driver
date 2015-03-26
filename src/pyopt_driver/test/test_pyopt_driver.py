import unittest

# pylint: disable=E0611,F0401
from nose import SkipTest

try:
    from pyopt_driver.pyopt_driver import pyOptDriver
except ImportError:
    pass

from openmdao.util.testutil import assert_rel_error
from openmdao.main.api import Assembly, set_as_top, Component, Driver
from openmdao.main.datatypes.api import Array, Float, Int
from openmdao.main.interfaces import IHasParameters, implements
from openmdao.main.hasparameters import HasParameters
from openmdao.util.decorators import add_delegate
from openmdao.examples.simple.paraboloid import Paraboloid
from openmdao.examples.simple.paraboloid_derivative import ParaboloidDerivative


class OptimizationUnconstrained(Assembly):
    """Unconstrained optimization of the Paraboloid with CONMIN."""

    def configure(self):
        """ Creates a new Assembly containing a Paraboloid and an optimizer"""

        # pylint: disable-msg=E1101

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

        self.driver.print_results = False


class OptimizationConstrained(Assembly):
    """Constrained optimization of the Paraboloid with CONMIN."""

    def configure(self):
        """ Creates a new Assembly containing a Paraboloid and an optimizer"""

        # pylint: disable=E1101

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


class ArrayParaboloid(Component):
    """ Evaluates the equation f(x) = (x[0]-3)^2 + x[0]x[1] + (x[1]+4)^2 - 3 """

    # pylint: disable=E1101
    x = Array([0., 0.], iotype='in', desc='The variable x')
    f_xy = Float(iotype='out', desc='F(x,y)')

    def execute(self):
        """Optimal solution (minimum): x = [6.6667,-7.3333]"""
        x = self.x
        self.f_xy = (x[0]-3.0)**2 + x[0]*x[1] + (x[1]+4.0)**2 - 3.0


class ArrayOpt(Assembly):
    """Constrained optimization of the ArrayParaboloid with CONMIN."""

    def configure(self):
        """ Creates a new Assembly containing ArrayParaboloid and an optimizer"""

        # pylint: disable=E1101

        self.add('paraboloid', ArrayParaboloid())
        self.add('driver', pyOptDriver())
        self.driver.pyopt_diff = True
        self.driver.workflow.add('paraboloid')
        self.driver.add_objective('paraboloid.f_xy')
        self.driver.add_parameter('paraboloid.x', low=-50., high=50.)
        self.driver.add_constraint('paraboloid.x[0]-paraboloid.x[1] >= 15.0')
        self.driver.print_results = False


class OptimizationConstrainedDerivatives(Assembly):
    """Constrained optimization of the Paraboloid with CONMIN."""

    def configure(self):
        """ Creates a new Assembly containing a Paraboloid and an optimizer"""

        # pylint: disable=E1101

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


class MultiFunction(Component):
    #Finds the minimum f(1) = x[1]
    #              and f(2) = (1+x[2])/x[1]

    # set up interface to the framework
    # pylint: disable=E1101
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

    def configure(self):
        """ Creates a new Assembly containing a MultiFunction and an optimizer"""

        # pylint: disable=E1101

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
        self.driver.add_parameter('multifunction.x1', low=0.1, high=1.0)
        self.driver.add_parameter('multifunction.x2', low=0.0, high=5.0)

        # NSGA2 Constraints
        self.driver.add_constraint('multifunction.g1_x >= 6.0')
        self.driver.add_constraint('multifunction.g2_x >= 1.0')


class BenchMark(Component):

    # set up interface to the framework
    # pylint: disable=E1101

    x1 = Int(10, iotype='in', desc='The variable x1')
    x2 = Int(10, iotype='in', desc='The variable x2')
    x3 = Int(10, iotype='in', desc='The variable x2')

    f_x = Float(iotype='out', desc='f(x)')

    g1_x = Float(iotype='out', desc='g(x)')
    h1_x = Float(iotype='out', desc='h(x)')

    def execute(self):

        x1 = self.x1
        x2 = self.x2
        x3 = self.x3

        #print "WAHHHH: ", self.x1

        self.f_x = -x1*x2*x3

        self.g1_x = x1 + 2.*x2 + 2.*x3 - 72.0
        self.h1_x = -x1 - 2.*x2 - 2.*x3


class BenchMarkOptimization(Assembly):
    """Benchmark Problem Objective optimization with ALPSO."""

    def __init__(self):
        """Creates a new Assembly containing a MultiFunction and an optimizer"""

        # pylint: disable=E1101

        super(BenchMarkOptimization, self).__init__()

        # Create MultiFunction component instances
        self.add('benchmark', BenchMark())

        # Create ALPSO Optimizer instance
        self.add('driver', pyOptDriver())

        # Driver process definition
        self.driver.workflow.add('benchmark')

        # PyOpt Flags
        self.driver.optimizer = 'ALPSO'
        self.driver.title = 'Bench mark problem 4 - Unconstrained'
        optdict = {}
        optdict['SwarmSize'] = 40
        optdict['maxOuterIter'] = 100
        optdict['maxInnerIter'] = 3
        optdict['minInnerIter'] = 3
        optdict['etol'] = 1e-4
        optdict['itol'] = 1e-4
        optdict['c1'] = 0.8
        optdict['c2'] = 0.8
        optdict['w1'] = 0.9
        optdict['nf'] = 5
        optdict['dt'] = 1.0
        optdict['vcrazy'] = 1e-4
        optdict['Scaling'] = 1
        optdict['seed'] = 1.0

        self.driver.options = optdict

        # ALPSO Objective
        self.driver.add_objective('benchmark.f_x')

        # ALPSO Design Variables
        self.driver.add_parameter('benchmark.x1', low=0, high=42)
        self.driver.add_parameter('benchmark.x2', low=0, high=42)
        self.driver.add_parameter('benchmark.x3', low=0, high=42)

        # ALPSO Constraints
        self.driver.add_constraint('benchmark.g1_x <= 0.0')
        self.driver.add_constraint('benchmark.h1_x <= 0.0')


class pyOptDriverTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        self.top = None

    def test_unconstrained(self):

        try:
            from pyopt_driver.pyopt_driver import pyOptDriver
        except ImportError:
            raise SkipTest("this test requires pyOpt to be installed")

        self.top = OptimizationUnconstrained()
        set_as_top(self.top)

        for optimizer in [ 'CONMIN', 'COBYLA', 'SNOPT', 'SLSQP' ] :

            try:
                self.top.driver.optimizer = optimizer
            except ValueError:
                raise SkipTest("%s not present on this system" % optimizer)

            self.top.driver.title = 'Little Test'
            optdict = {}
            self.top.driver.options = optdict
            self.top.driver.pyopt_diff = True

            self.top.run()

            assert_rel_error(self, self.top.paraboloid.x, 6.6667, 0.01)
            assert_rel_error(self, self.top.paraboloid.y, -7.3333, 0.01)


    def test_basic_CONMIN(self):

        try:
            from pyopt_driver.pyopt_driver import pyOptDriver
        except ImportError:
            raise SkipTest("this test requires pyOpt to be installed")

        self.top = OptimizationConstrained()
        set_as_top(self.top)

        try:
            self.top.driver.optimizer = 'CONMIN'
            self.top.driver.optimizer = 'COBYLA'
        except ValueError:
            raise SkipTest("CONMIN not present on this system")

        self.top.driver.title = 'Little Test'
        optdict = {}
        self.top.driver.options = optdict
        self.top.driver.pyopt_diff = True

        self.top.run()

        assert_rel_error(self, self.top.paraboloid.x, 7.175775, 0.01)
        assert_rel_error(self, self.top.paraboloid.y, -7.824225, 0.01)

    def test_array_CONMIN(self):

        try:
            from pyopt_driver.pyopt_driver import pyOptDriver
        except ImportError:
            raise SkipTest("this test requires pyOpt to be installed")

        self.top = ArrayOpt()
        set_as_top(self.top)

        try:
            self.top.driver.optimizer = 'CONMIN'
        except ValueError:
            raise SkipTest("CONMIN not present on this system")

        self.top.driver.title = 'Little Test'
        optdict = {}
        self.top.driver.options = optdict

        self.top.run()  # Run with pyopt finite differencing.

        assert_rel_error(self, self.top.paraboloid.x[0], 7.175775, 0.01)
        assert_rel_error(self, self.top.paraboloid.x[1], -7.824225, 0.01)

        # Re-run with OpenMDAO derivatives.
#        self.top.paraboloid.x = [0., 0.]
#        self.top.driver.pyopt_diff = False
#        self.top.run()
#        assert_rel_error(self, self.top.paraboloid.x[0], 7.175775, 0.01)
#        assert_rel_error(self, self.top.paraboloid.x[1], -7.824225, 0.01)

    def test_basic_CONMIN_derivatives(self):

        try:
            from pyopt_driver.pyopt_driver import pyOptDriver
        except ImportError:
            raise SkipTest("this test requires pyOpt to be installed")

        self.top = OptimizationConstrainedDerivatives()
        set_as_top(self.top)

        try:
            self.top.driver.optimizer = 'CONMIN'
        except ValueError:
            raise SkipTest("CONMIN not present on this system")

        self.top.driver.title = 'Little Test with Gradient'
        optdict = {}
        self.top.driver.options = optdict

        self.top.run()

        assert_rel_error(self, self.top.paraboloid.x, 7.175775, 0.01)
        assert_rel_error(self, self.top.paraboloid.y, -7.824225, 0.01)

    def test_GA_multi_obj_multi_con(self):
        # Note, just verifying that things work functionally, rather than run
        # this for many generations.

        try:
            from pyopt_driver.pyopt_driver import pyOptDriver
        except ImportError:
            raise SkipTest("this test requires pyOpt to be installed")

        self.top = MultiObjectiveOptimization()
        set_as_top(self.top)

        try:
            self.top.driver.optimizer = 'NSGA2'
        except ValueError:
            raise SkipTest("NSGA2 not present on this system")

        # PyOpt Flags
        self.top.driver.title = 'Two-Objective Fitness MultiFunction Test'
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

    def test_ALPSO_integer_design_var(self):

        #    probNEW.py
        #
        #Set's up component for Schittkowski's TP37 Problem.
        #
        #    min 	-x1*x2*x3
        #    s.t.:	x1 + 2.*x2 + 2.*x3 - 72 <= 0
        #            - x1 - 2.*x2 - 2.*x3 <= 0
        #            0 <= xi <= 42,  i = 1,2,3
        #
        #    f* = -3456 , x* = [24, 12, 12]
        #
        # *Problem taken from pyOpt example tp037

        try:
            from pyopt_driver.pyopt_driver import pyOptDriver
        except ImportError:
            raise SkipTest("this test requires pyOpt to be installed")

        opt_problem = BenchMarkOptimization()
        set_as_top(opt_problem)
        opt_problem.run()

        self.assertEqual(opt_problem.benchmark.x1, 24)
        self.assertEqual(opt_problem.benchmark.x2, 12)
        self.assertEqual(opt_problem.benchmark.x3, 12)


    def test_initial_run(self):
        # Test to make sure fix that put run_iteration
        #   at the top of the execute method is in place and working
        class MyComp(Component):

            x = Float(0.0, iotype='in', low=-10, high=10)
            xx = Float(0.0, iotype='in', low=-10, high=10)
            f_x = Float(iotype='out')
            y = Float(iotype='out')

            def execute(self):
                if self.xx != 1.0:
                    self.raise_exception("Lazy", RuntimeError)
                self.f_x = 2.0*self.x
                self.y = self.x

        @add_delegate(HasParameters)
        class SpecialDriver(Driver):

            implements(IHasParameters)

            def execute(self):
                self.set_parameters([1.0])

        top = set_as_top(Assembly())
        top.add('comp', MyComp())

        try:
            from pyopt_driver.pyopt_driver import pyOptDriver
        except ImportError:
            raise SkipTest("this test requires pyOpt to be installed")

        try:
            top.driver.optimizer = 'CONMIN'
        except ValueError:
            raise SkipTest("CONMIN not present on this system")

        top.driver.title = 'Little Test'
        optdict = {}
        top.driver.options = optdict
        top.driver.pyopt_diff = True

        top.add('driver', pyOptDriver())
        top.add('subdriver', SpecialDriver())
        top.driver.workflow.add('subdriver')
        top.subdriver.workflow.add('comp')

        top.subdriver.add_parameter('comp.xx')
        top.driver.add_parameter('comp.x')
        top.driver.add_constraint('comp.y > 1.0')
        top.driver.add_objective('comp.f_x')

        top.run()

if __name__ == "__main__":
    unittest.main()

