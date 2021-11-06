- HTM - implementation of spacial 
pooler which is optimised for inputs so sparse that only a handful of minicolumns have 
any connection to any of the active inputs
- HTM2 - implementation of spacial pooler that iterates all minicolumns, assuming that
most likely most minicolumns will have at least one connection to some active input
- HTM3 - spacial pooler that increments/decrements its permanences with
a certain momentum. This makes learning more stable and most importantly, disconnects
  all synapses that convey more random noise than actual signal.
  For example if a certain input is randomly active 50% of the time,
  then the "standard" implementation of spacial pooler would average-out
  all increments and decrements and as a result, the permanence would not 
  change much. We could force such noisy synapses to disconnect if we set 
  permanence_decrement value larger than permanence_increment. However,
  this leads to another problem. Suppose the same input is always active
  in response to two observations.
- HTM4 - implements HTM2 with addition of negative synapses
- HTM5 - implements a mix of HTM3 and HTM4. Has both negative synapses and momentum

