"""
OpenMDAO Wrapper for pyOpt.

pyOpt is an object-oriented framework for formulating and solving nonlinear
constrained optimization problems.
"""

# pylint: disable-msg=E0611,F0401
from numpy import array, float32, float64, int32, int64, zeros

from pyOpt import Optimization
        
from openmdao.lib.datatypes.api import Bool, Dict, Enum, Str
from openmdao.main.api import Driver
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
    print_results = Bool(True, iotype = 'in', 
                         desc='Print pyOpt results if True')
    pyopt_diff = Bool(False, iotype='in', desc='Set to True to let pyOpt'
                       'calculate the gradient.')
    
    def __init__(self):
        """Initialize pyopt - not much needed."""
        
        super(pyOptDriver, self).__init__()
        
        self.pyOpt_solution = None
        self.param_type = {}
    
    def execute(self):
        """pyOpt execution. Note that pyOpt controls the execution, and the
        individual optimizers control the iteration."""
        
        self.pyOpt_solution = None
    
        opt_prob = Optimization(self.title, self.objfunc, var_set={}, 
                                obj_set={}, con_set={})
        
        # Add all parameters
        self.param_type = {}
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
            self.param_type[name] = vartype

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
        if self.pyopt_diff:
            # Use pyOpt's internal finite difference
            opt(opt_prob, sens_type='FD')
        else:
            # Use OpenMDAO's differentiator for the gradient
            opt(opt_prob, sens_type=self.gradfunc)
        
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
            
            # Note: Sometimes pyOpt sends us an x array that is larger than
            # the number of parameters. In the pyOpt examples, they just take
            # the first n entries as the parameters, so we do too.
            nparam = len(self.param_type)
            
            # Integer parameters come back as floats, so we need to round them
            # and turn them into python integers before setting.
            if 'i' in self.param_type.values():
                for j, param in enumerate(self.get_parameters().keys()):
                    if self.param_type[param] == 'i':
                        self.set_parameter_by_name(param, int(round(x[j])))
                    else:
                        self.set_parameter_by_name(param, x[j])
            else:
                self.set_parameters([val for val in x[0:nparam]])
            
            # Execute the model
            self.run_iteration()
            
            # Get the objective function evaluations
            for obj in self.eval_objectives():
                f.append(obj)
                
            f = array(f)
            
            # Constraints. Note that SLSQP defines positive as satisfied.
            con_list = []
            if len(self.get_eq_constraints()) > 0 :
                con_list.extend([v.evaluate(self.parent) for \
                                v in self.get_eq_constraints().values()])
            if len(self.get_ineq_constraints()) > 0 :
                con_list.extend([v.evaluate(self.parent) for \
                                v in self.get_ineq_constraints().values()])
                
            g = array(con_list)
                
            # Print out cases whenever the objective function is evaluated.
            # TODO: pyOpt's History object might be better suited, though
            # it does not seem to be part of the Optimization object at
            # present.
            self.record_case()
            
        except Exception, msg:
            
            # Exceptions seem to be swallowed by the C code, so this
            # should give the user more info than the dreaded "segfault"
            print "Exception: %s" % str(msg)
            print 70*"="
            import traceback
            traceback.print_exc()
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
        
        try:
            inputs = self.get_parameters().keys()
            obj = ["%s.out0" % item.pcomp_name for item in \
                   self.get_objectives().values()]
            econ = ["%s.out0" % item.pcomp_name for item in \
                    self.get_eq_constraints().values()]
            icon = ["%s.out0" % item.pcomp_name for item in \
                    self.get_ineq_constraints().values()]
    
            J = self.workflow.calc_gradient(inputs, obj + econ + icon)
            
            nobj = len(obj)
            ncon = len(econ) + len(icon)
            nparam = len(inputs)
            
            df = zeros([nobj, nparam])
            dg = zeros([ncon, nparam])
            
            df = J[0:nobj, :]
            
            n1 = nobj
            n2 = nobj + ncon
            if ncon > 0:
                dg = -J[n1:n2, :]

        except Exception, msg:

            # Exceptions seem to be swallowed by the C code, so this
            # should give the user more info than the dreaded "segfault"
            print "Exception: %s" % str(msg)
            return df, dg, fail
        
        fail = 0
        
        return df, dg, fail

