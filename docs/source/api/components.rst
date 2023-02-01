.. module:: popkinmocks.components

.. _api:

Components
=================

Components classes are related via the inheritance diagram,

.. inheritance-diagram:: FromParticle Stream GrowingDisk Mixture
   :top-classes: Component
   :parts: 1

For each subclass, this API lists public methods are new or newly re-implemted
for that subclass e.g. a docstring for `evaluate_ybar` appears for `Component`, 
`ParametricComponent` and `Mixture` as they have their own implementations,
while it does not appear for `FromParticle` which inherits `evaluate_ybar`
from `Component`.

`Component`
-----------------

.. autoclass:: Component
   :members:

`FromParticle`
----------------------

.. autoclass:: FromParticle
    :members:

`ParametricComponent`
-----------------------

.. autoclass:: ParametricComponent
    :members:

`GrowingDisk`
^^^^^^^^^^^^^

.. autoclass:: GrowingDisk
    :members:

`Stream`
^^^^^^^^

.. autoclass:: Stream
    :members:

`Mixture`
-----------------------

.. autoclass:: Mixture
    :members: