
================
Package Metadata
================

- **author:** Kenneth T. Moore

- **author-email:** kenneth.t.moore-1@nasa.gov

- **classifier**:: 

    Intended Audience :: Science/Research
    Topic :: Scientific/Engineering

- **description-file:** README.txt

- **entry_points**:: 

    [openmdao.component]
    test_pyopt_driver.OptimizationConstrained=test_pyopt_driver:OptimizationConstrained
    test_pyopt_driver.OptimizationConstrainedDerivatives=test_pyopt_driver:OptimizationConstrainedDerivatives
    pyopt_driver.pyopt_driver.pyOptDriver=pyopt_driver.pyopt_driver:pyOptDriver
    test_pyopt_driver.MultiObjectiveOptimization=test_pyopt_driver:MultiObjectiveOptimization
    test_pyopt_driver.MultiFunction=test_pyopt_driver:MultiFunction
    [openmdao.driver]
    pyopt_driver.pyopt_driver.pyOptDriver=pyopt_driver.pyopt_driver:pyOptDriver
    [openmdao.container]
    test_pyopt_driver.OptimizationConstrained=test_pyopt_driver:OptimizationConstrained
    test_pyopt_driver.OptimizationConstrainedDerivatives=test_pyopt_driver:OptimizationConstrainedDerivatives
    pyopt_driver.pyopt_driver.pyOptDriver=pyopt_driver.pyopt_driver:pyOptDriver
    test_pyopt_driver.MultiObjectiveOptimization=test_pyopt_driver:MultiObjectiveOptimization
    test_pyopt_driver.MultiFunction=test_pyopt_driver:MultiFunction

- **home-page:** https://github.com/OpenMDAO-Plugins/pyopt-driver

- **keywords:** openmdao

- **license:** Apache License, Version 2.0

- **maintainer:** Kenneth T. Moore

- **maintainer-email:** kenneth.t.moore-1@nasa.gov

- **name:** pyopt_driver

- **requires-dist:** openmdao.main

- **requires-python**:: 

    >=2.6
    <3.0

- **static_path:** [ '_static' ]

- **summary:** OpenMDAO driver wrapper for the open-source optimization package pyOpt

- **version:** 0.8

