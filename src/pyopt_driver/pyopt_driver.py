"""
OpenMDAO Wrapper for pyOpt.

pyOpt is an object-oriented framework for formulating and solving nonlinear
constrained optimization problems.
"""

# pylint: disable-msg=E0611,F0401
from numpy import array, float32, float64, int32, int64, zeros

from pyOpt import Optimization
        
from openmdao.lib.datatypes.api import Bool, Dict, Enum, Str
from openmdao.main.api import DriverUsesDerivatives
from openmdao.main.interfaces import IHasParameters, IHasConstraints, \
                                     IHasObjective, implements, IOptimizer
from openmdao.main.hasparameters import HasParameters
from openmdao.main.hasconstraints import HasConstraints
from openmdao.main.hasobjective import HasObjectives
from openmdao.util.decorators import add_delegate

def _check_imports():
    """ Dynamically remove optimizers we don't have
    """
    
    optlist = ['ALHSO', 'ALPSO', 'COBYLA', 'CONMIN', 'FSQP', 'GCMMA', 'KSOPT',
               'MIDACO', 'MMA', 'MMFD', 'NLPQL', 'NSGA2', 'PSQP', 'SLSQP', 
               'SNOPT', 'SOLVOPT']

    for optimizer in optlist:
        try:
            exec('from pyOpt import %s' % optimizer)
        except ImportError:
            optlist.remove(optimizer)
            
    return optlist
    
@add_delegate(HasParameters, HasConstraints, HasObjectives)
class pyOptDriver(DriverUsesDerivatives):
    """ Driver wrapper for pyOpt. 
    """

    implements(IHasParameters, IHasConstraints, IHasObjective, IOptimizer)
    
    optimizer = Enum('ALPSO', _check_imports(), iotype='in', 
                       desc='Name of optimizers to use')
    title = Str('Optimization using pyOpt', iotype='in', 
                desc='Title of this optimization run')
    options = Dict(iotype='in', 
                   desc='Dictionary of optimization parameters')
    print_results = Bool(True, iotype = 'in', 
                         desc='Print pyOpt results if True')
    
    def __init__(self):
        """Initialize pyopt - not much needed."""
        
        super(pyOptDriver, self).__init__()
        
        self.pyOpt_solution = None
    
    def execute(self):
        """pyOpt execution. Note that pyOpt controls the execution, and the
        individual optimizers control the iteration."""
        
        self.pyOpt_solution = None
    
        opt_prob = Optimization(self.title, self.objfunc, var_set={}, 
                                obj_set={}, con_set={})
        
        # Add all parameters
        for name, param in self.get_parameters().iteritems():
            
            val = param.evaluate()
            
            # We need to identify Enums, Lists, Dicts
            metadata = param.get_metadata()[0][1]          
            
            # enumerated, discrete or continuous
            choices = []
            if ('values' in metadata and \
               isinstance(metadata['values'],(list, tuple, array, set))):
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
                msg = 'Only continuous, descrete, or enumerated variables ' + \
                      'are supported. %s is %s.' % (name, type(val))
                self.raise_exception(msg, ValueError)
            
            opt_prob.addVar(name, vartype, lower=param.low, upper=param.high, 
                            value=val, choices=choices)

        # Add all objectives
        for name in self.get_objectives().keys():
            opt_prob.addObj(name)
            
        # Add all equality constraints
        for name in self.get_eq_constraints().keys():
            opt_prob.addCon(name, type='e')
        
        # Add all inequality constraints
        for name in self.get_ineq_constraints().keys():
            opt_prob.addCon(name, type='i')

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
        if self.differentiator:
            # Use OpenMDAO's differentiator for the gradient
            opt(opt_prob, sens_type=self.gradfunc)
        else:
            # Use pyOpt's internal finite difference
            opt(opt_prob, sens_type='FD')
        
        # Print results
        if self.print_results:
            print opt_prob.solution(0)
        
        # Pull optimal parameters back into framework and re-run, so that
        # framework is left in the right final state
        dvals = []
        for i in range(0, len(opt_prob.solution(0)._variables)):
            dvals.append(opt_prob.solution(0)._variables[i].value)
        self.set_parameters(dvals)
        self.run_iteration()
        self.record_case()
        
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
            
            # Note: Sometimes pyOpt sends us an x array that is larger than the 
            # number of parameters. In the pyOpt examples, they just take the first
            # n entries as the parameters, so we do too.
            nparam = self.get_parameters().__len__()
            self.set_parameters([val for val in x[0:nparam]])
            
            # Execute the model
            self.run_iteration()
            
            # Get the objective function evaluations
            for obj in self.eval_objectives():
                f.append(obj)
                
            f = array(f)
            
            # Get the constraint evaluations
            for con in self.get_eq_constraints().values():
                g.append(con.evaluate(self.parent)[0])
            
            for con in self.get_ineq_constraints().values():
                val = con.evaluate(self.parent)
                if '>' in val[2]:
                    g.append(val[1]-val[0])
                else:
                    g.append(val[0]-val[1])
                    
            g = array(g)
            
            # Print out cases whenever the objective function is evaluated.
            # TODO: pyOpt's History object might be better suited, though
            # it does not seem to be part of the Optimization object at
            # present.
            self.record_case()
            
        except Exception, msg:
            
            # Exceptions seem to be swallowed by the C code, so this
            # should give the user more info than the dreaded "segfault"
            print "Exception: %s" % str(msg)
            return f, g, fail
        
        fail = 0
        
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
            # Keys are used to conveniently access the gradient
            param_names = self.get_parameters().keys()
            n_param = len(param_names)
            obj_names = self.get_objectives().keys()
            n_obj = len(obj_names)
            con_names = list(self.get_eq_constraints().keys() + \
                           self.get_ineq_constraints().keys())
            n_con = len(con_names)

            df = zeros([n_obj, n_param])
            dg = zeros([n_con, n_param])
            
            # Calculate the gradient. Fake finite difference
            # is supported using the FiniteDifference differentiator.
            self.ffd_order = 1
            self.differentiator.calc_gradient()
            self.ffd_order = 0
    
            i = 0
            for name in obj_names:
                df[i][:] = self.differentiator.get_gradient(name)
                i += 1
            
            i = 0
            for param_name in param_names:
                j = 0
                for con_name in con_names:
                    dg[j][i] = self.differentiator.get_derivative(con_name,
                                                               wrt=param_name)
                    j += 1 
                i += 1 

        except Exception, msg:

            # Exceptions seem to be swallowed by the C code, so this
            # should give the user more info than the dreaded "segfault"
            print "Exception: %s" % str(msg)
            return df, dg, fail
        
        fail = 0
        
        return df, dg, fail

if __name__ == "__main__": # pragma: no cover         

    from openmdao.main.api import set_as_top
    from openmdao.main.api import Assembly, set_as_top
    from openmdao.lib.differentiators.finite_difference import FiniteDifference
    from openmdao.examples.simple.paraboloid_derivative import ParaboloidDerivative
