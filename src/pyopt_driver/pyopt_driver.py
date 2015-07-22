"""
OpenMDAO Wrapper for pyOpt.

pyOpt is an object-oriented framework for formulating and solving nonlinear
constrained optimization problems.
"""

# pylint: disable=E0611,F0401
from numpy import array, float32, float64, int32, int64

from pyOpt import Optimization

from openmdao.main.api import Driver
from openmdao.main.datatypes.api import Bool, Dict, Enum, Str
from openmdao.main.interfaces import IHasParameters, IHasConstraints, \
                                     IHasObjective, implements, IOptimizer
from openmdao.main.hasparameters import HasParameters
from openmdao.main.hasconstraints import HasConstraints
from openmdao.main.hasobjective import HasObjectives
from openmdao.util.decorators import add_delegate


def _check_imports():
    """ Dynamically remove optimizers we don't have
    """

    optlist = ['ALGENCIAN', 'ALHSO', 'ALPSO', 'COBYLA', 'CONMIN', 'FILTERSD',
               'FSQP', 'GCMMA', 'KSOPT', 'MIDACO', 'MMA', 'MMFD', 'NLPQL', 'NLPQLP',
               'NSGA2', 'PSQP', 'SDPEN', 'SLSQP', 'SNOPT', 'SOLVOPT']

    for optimizer in optlist[:]:
        try:
            exec('from pyOpt import %s' % optimizer)
        except ImportError:
            optlist.remove(optimizer)

    return optlist


@add_delegate(HasParameters, HasConstraints, HasObjectives)
class pyOptDriver(Driver):
    """ Driver wrapper for pyOpt.
    """

    implements(IHasParameters, IHasConstraints, IHasObjective, IOptimizer)

    optimizer = Enum('ALPSO', _check_imports(), iotype='in',
                     desc='Name of optimizers to use')
    title = Str('Optimization using pyOpt', iotype='in',
                desc='Title of this optimization run')
    options = Dict(iotype='in',
                   desc='Dictionary of optimization parameters')
    print_results = Bool(True, iotype='in',
                         desc='Print pyOpt results if True')
    pyopt_diff = Bool(False, iotype='in',
                      desc='Set to True to let pyOpt calculate the gradient')
    store_hst = Bool(False, iotype='in',
                     desc='Store optimization history if True')
    hot_start = Bool(False, iotype='in',
                     desc='resume optimization run using stored history if True')

    def __init__(self):
        """Initialize pyopt - not much needed."""

        super(pyOptDriver, self).__init__()

        self.pyOpt_solution = None
        self.param_type = {}
        self.nparam = None

        self.inputs = None
        self.objs = None
        self.cons = None

    def execute(self):
        """pyOpt execution. Note that pyOpt controls the execution, and the
        individual optimizers control the iteration."""

        self.pyOpt_solution = None

        self.run_iteration()

        opt_prob = Optimization(self.title, self.objfunc, var_set={},
                                obj_set={}, con_set={})

        # Add all parameters
        self.param_type = {}
        self.nparam = self.total_parameters()
        for name, param in self.get_parameters().iteritems():

            # We need to identify Enums, Lists, Dicts
            metadata = param.get_metadata()[1]
            values = param.evaluate()

            # Assuming uniform enumerated, discrete, or continuous for now.
            val = values[0]
            choices = []
            if 'values' in metadata and \
               isinstance(metadata['values'], (list, tuple, array, set)):
                vartype = 'd'
                choices = metadata['values']
            elif isinstance(val, bool):
                vartype = 'd'
                choices = [True, False]
            elif isinstance(val, (int, int32, int64)):
                vartype = 'i'
            elif isinstance(val, (float, float32, float64)):
                vartype = 'c'
            else:
                msg = 'Only continuous, discrete, or enumerated variables' \
                      ' are supported. %s is %s.' % (name, type(val))
                self.raise_exception(msg, ValueError)
            self.param_type[name] = vartype

            names = param.names
            lower_bounds = param.get_low()
            upper_bounds = param.get_high()
            for i in range(param.size):
                opt_prob.addVar(names[i], vartype,
                                lower=lower_bounds[i], upper=upper_bounds[i],
                                value=values[i], choices=choices)
        # Add all objectives
        for name in self.get_objectives():
            opt_prob.addObj(name)

        # Add all equality constraints
        for name, con in self.get_eq_constraints().items():
            if con.size > 1:
                for i in range(con.size):
                    opt_prob.addCon('%s [%s]' % (name, i), type='e')
            else:
                opt_prob.addCon(name, type='e')

        # Add all inequality constraints
        for name, con in self.get_ineq_constraints().items():
            if con.size > 1:
                for i in range(con.size):
                    opt_prob.addCon('%s [%s]' % (name, i), type='i')
            else:
                opt_prob.addCon(name, type='i')

        self.inputs = self.list_param_group_targets()
        self.objs = self.list_objective_targets()
        self.cons = self.list_constraint_targets()

        # Instantiate the requested optimizer
        optimizer = self.optimizer
        try:
            exec('from pyOpt import %s' % optimizer)
        except ImportError:
            msg = "Optimizer %s is not available in this installation." % \
                   optimizer
            self.raise_exception(msg, ImportError)

        optname = vars()[optimizer]
        opt = optname()

        # Set optimization options
        for option, value in self.options.iteritems():
            opt.setOption(option, value)

        # Execute the optimization problem
        if self.pyopt_diff:
            # Use pyOpt's internal finite difference
            opt(opt_prob, sens_type='FD', sens_step=self.gradient_options.fd_step,
                store_hst=self.store_hst, hot_start=self.hot_start)
        else:
            # Use OpenMDAO's differentiator for the gradient
            opt(opt_prob, sens_type=self.gradfunc, store_hst=self.store_hst,
                hot_start=self.hot_start)

        # Print results
        if self.print_results:
            print opt_prob.solution(0)

        # Pull optimal parameters back into framework and re-run, so that
        # framework is left in the right final state
        dvals = []
        for i in range(0, len(opt_prob.solution(0)._variables)):
            dvals.append(opt_prob.solution(0)._variables[i].value)

        # Integer parameters come back as floats, so we need to round them
        # and turn them into python integers before setting.
        if 'i' in self.param_type.values():
            for j, param in enumerate(self.get_parameters().keys()):
                if self.param_type[param] == 'i':
                    dvals[j] = int(round(dvals[j]))

        self.set_parameters(dvals)
        self.run_iteration()

        # Save the most recent solution.
        self.pyOpt_solution = opt_prob.solution(0)

    def objfunc(self, x, *args, **kwargs):
        """ Function that evaluates and returns the objective function and
        constraints. This function is passed to pyOpt's Optimization object
        and is called from its optimizers.

        x: array
            Design variables

        args and kwargs are also passed in, but aren't used.

        Returns

        f: array
            Objective function evaluated at design variables

        g: array
            Constraints evaluated at design variables

        fail: int
            0 for successful function evaluation
            1 for unsuccessful function evaluation
        """

        fail = 1
        f = []
        g = []

        try:

            # Note: Sometimes pyOpt sends us an x array that is larger than
            # the number of parameters. In the pyOpt examples, they just take
            # the first n entries as the parameters, so we do too.

            # Integer parameters come back as floats, so we need to round them
            # and turn them into python integers before setting.
            param_types = self.param_type
            if 'i' in param_types.values():
                j = 0
                for name, param in self.get_parameters().iteritems():
                    size = param.size
                    if param_types[name] == 'i':
                        self.set_parameter_by_name(name, int(round(x[j:j+size])))
                    else:
                        self.set_parameter_by_name(name, x[j:j+size])
                    j += size
            else:
                self.set_parameters(x[0:self.nparam])

            # Execute the model
            self.run_iteration()

            # Get the objective function evaluations
            f = array(self.eval_objectives())

            # Get the constraint evaluations
            g = array(self.eval_constraints(self.parent)).tolist()

            fail = 0

        except Exception as msg:

            # Exceptions seem to be swallowed by the C code, so this
            # should give the user more info than the dreaded "segfault"
            print "Exception: %s" % str(msg)
            print 70*"="
            import traceback
            traceback.print_exc()
            print 70*"="

        return f, g, fail

    def gradfunc(self, x, f, g, *args, **kwargs):
        """ Function that evaluates and returns the gradient of the objective
        function and constraints. This function is passed to pyOpt's
        Optimization object and is called from its optimizers.

        x: array
            Design variables

        f: array
            Objective function evaluated at design variables
            Note: unneeded in OpenMDAO, so unused

        g: array
            Constraints evaluated at design variables
            Note: unneeded in OpenMDAO, so unused

        args and kwargs are also passed in, but aren't used.

        Returns

        d_obj: array
            Gradient of the objective

        d_con: array
            Gradient of the constraints

        fail: int
            0 for successful function evaluation
            1 for unsuccessful function evaluation
        """

        fail = 1
        df = []
        dg = []

        try:
            J = self.workflow.calc_gradient(self.inputs, self.objs + self.cons)

            nobj = len(self.objs)
            df = J[0:nobj, :]
            dg = J[nobj:, :]

            fail = 0

        except Exception as msg:

            # Exceptions seem to be swallowed by the C code, so this
            # should give the user more info than the dreaded "segfault"
            print "Exception: %s" % str(msg)
            print 70*"="
            import traceback
            traceback.print_exc()
            print 70*"="

        return df, dg, fail

    def requires_derivs(self):
        return True
