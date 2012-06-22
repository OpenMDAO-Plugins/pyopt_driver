

===========
Usage Guide
===========

This is the OpenMDAO wrapper for pyOpt. Before installing this package, pyOpt
must be installed to either your system level Python or your local Python
environment in OpenMDAO. Please visit http://www.pyopt.org to download and
learn more about pyOpt.

This wrapper should work with all of the optimizers included in pyOpt. Some of
these optimizers are commercial products, which won't be available if you
don't already have them, but there are still seven or eight optimizers that are public
domain or open source.

The pyOpt driver behaves like any other optimizer driver in OpenMDAO. As such,
it can optimize any workflow that includes any combination of assemblies,
components, and drivers. Keep in mind that this is a general optimization
package, so the driver interface will allow you to, for example, add two 
objectives to the problem even if you've selected the CONMIN optimizer. So exercise
care and make sure the optimizer you choose can handle your problem in
terms of number of objectives, support for equality constraints, support for
inequality constraints, and support for integer or enumerated parameters.

Here is a simple example where ALPSO (Augmented Lagrangian Particle Swarm
Optimizer) is used to minimize the constrained paraboloid problem from the
OpenMDAO examples.

.. testcode:: pyOpt_basic

        from pyopt_driver.pyopt_driver import pyOptDriver
        
        from openmdao.main.api import Assembly
        from openmdao.examples.simple.paraboloid import Paraboloid
        
        class OptimizationConstrained(Assembly):
            """Constrained optimization of a Paraboloid."""
            
            def configure(self):
                """ Creates a new Assembly containing a Paraboloid and an optimizer"""
                
                # Create Paraboloid component instances
                self.add('paraboloid', Paraboloid())
        
                # Create pyOpt driver instance
                self.add('driver', pyOptDriver())
                
                # Driver process definition
                self.driver.workflow.add('paraboloid')
                
                # PyOpt Flags
                self.driver.optimizer = 'ALPSO'
                self.driver.title='Simple Test'
                self.driver.print_results = True
                optdict = {}
                optdict['SwarmSize'] = 30
                optdict['etol'] = 1e-3
                self.driver.options = optdict
                        
                # Objective 
                self.driver.add_objective('paraboloid.f_xy')
                
                # Design Variables 
                self.driver.add_parameter('paraboloid.x', low=-50., high=50.)
                self.driver.add_parameter('paraboloid.y', low=-50., high=50.)
                
                # Constraints
                self.driver.add_constraint('paraboloid.x-paraboloid.y >= 15.0')
                
                
        if __name__ == "__main__": # pragma: no cover         
        
            import time
            from openmdao.main.api import set_as_top
            
            opt_problem = OptimizationConstrained()
            set_as_top(opt_problem)
            
            tt = time.time()
            opt_problem.run()
        
            print "\n"
            print "Minimum found at (%f, %f)" % (opt_problem.paraboloid.x, \
                                                 opt_problem.paraboloid.y)
            print "Elapsed time: ", time.time()-tt, "seconds"

The pyOpt wrapper contains a variable `optimizer` where the optimizer name can
be specified. This variable is an `Enum` that contains all of the valid optimizers
in the pyOpt installation. This list is determined when the wrapper component is
instantiated, so it always holds the most accurate list of what optimizers are
available.

The `title` variable can be used to give the solution a title, which shows up in
the pyOpt output. The ``print_results`` controls printing of pyOpt's solution object.
Its default is ``True``, which means results are always printed.

Additionally, each optimizer has its own specialized settings that can be changed 
using the `options` variable, which is a dictionary that can contain a setting
name as the `key` and a new setting value as the `value`. A list of the 
available settings for each optimizer should be found in the pyOpt documentation. In
this example, we set the swarm size and the absolute tolerance for equality constraints.

After the pyOpt driver is executed, the driver's workflow is left in the
optimal state that the optimizer determined. If you would like to access the
solution object that pyOpt generates, it's in the attribute ``pyOpt_solution``.

When a gradient optimizer is used, pyOpt calculates the gradient using its internal
finite difference. You can also use an OpenMDAO differentiator by inserting it into
the differentiator slot.
