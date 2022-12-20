.. OpenBox documentation master file, created by
   sphinx-quickstart on Fri Dec 23 13:31:47 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/PKU-DAIR/open-box

###############################################################
This is a docs for test!!! :octicon:`rocket;1em`
###############################################################

**OpenBox** is an efficient open-source system designed for **solving
generalized black-box optimization (BBO) problems**, such as
`automatic hyper-parameter tuning <./examples/single_objective_hpo.html>`__,
automatic A/B testing, experimental design, database knob tuning,
processor architecture and circuit design,
resource allocation, automatic chemical design, etc.

The design of **OpenBox** follows the philosophy of providing **"BBO as a service"** - we
opt to implement **OpenBox** as a distributed, fault-tolerant, scalable, and efficient service,
with a wide range of application scope, stable performance across problems
and advantages such as ease of use, portability, and zero maintenance.

There are two ways to use **OpenBox**:
`Standalone python package <./installation/installation_guide.html>`__
and `Online BBO service <./openbox_as_service/service_introduction.html>`__.

**OpenBox GitHub:** https://github.com/PKU-DAIR/open-box

------------------------------------------------

.. include:: ../../README.md
   :parser: myst_parser.sphinx_
   :start-after: <!-- start of news (for docs) -->
   :end-before: <!-- end of news (for docs) -->

------------------------------------------------

Who should consider using OpenBox
---------------------------------

-  Those who want to **tune hyper-parameters** for their ML tasks
   automatically.

-  Those who want to **find the optimal configuration** for their
   configuration search tasks (e.g., database knob tuning).

-  Data platform owners who want to **provide BBO service in their
   platform**.

-  Researchers and data scientists who want to **solve
   generalized BBO problems easily**.

------------------------------------------------

.. _openbox-characteristics--capabilities:

OpenBox capabilities
--------------------------------------

OpenBox has a wide range of functionality scope, which includes:

1. BBO with multiple objectives and constraints.

2. BBO with transfer learning.

3. BBO with distributed parallelization.

4. BBO with multi-fidelity acceleration.

5. BBO with early stops.

In the following, we provide a taxonomy of existing BBO systems:

============== ========== ==== ========== ======= ===========
System/Package Multi-obj. FIOC Constraint History Distributed
============== ========== ==== ========== ======= ===========
Hyperopt       ×          √    ×          ×       √
Spearmint      ×          ×    √          ×       ×
SMAC3          ×          √    ×          ×       ×
BoTorch        √          ×    √          ×       ×
GPflowOPT      √          ×    √          ×       ×
Vizier         ×          √    ×          △       √
HyperMapper    √          √    √          ×       ×
HpBandSter     ×          √    ×          ×       √
**OpenBox**    √          √    √          √       √
============== ========== ==== ========== ======= ===========

-  **FIOC**: Support different input variable types, including
   Float, Integer, Ordinal and Categorical.

-  **Multi-obj.**: Support optimizing multiple objectives.

-  **Constraint**: Support inequality constraints.

-  **History**: Support injecting prior knowledge from previous
   tasks into the current search. (△ means the system cannot support
   it for general cases)

-  **Distributed**: Support parallel evaluations in a distributed
   environment.

------------------------------------------------

..  toctree::
    :caption: This is a docs for TEST!!!
    :maxdepth: 2
    :titlesonly:

    test md <test_md>
    Overview <overview/overview>
    Installation <installation/installation_guide>
    Quick Start :octicon:`rocket;1em` <quick_start/quick_start>
    Examples <examples/examples>
    Visualization (New!) <visualization/visualization>
    Advanced Usage <advanced_usage/advanced_usage>
    OpenBox as Service <openbox_as_service/openbox_as_service>
    Research and Publications <research_and_publications/research_and_publications>
    Change Logs <change_logs/change_logs>
