.. _engine-plumed:

PLUMED
======


.. list-table::
   :header-rows: 1

   * - Official website
     - How is metatensor supported?
   * - https://www.plumed.org/
     - In the official (development) version


Supported model outputs
^^^^^^^^^^^^^^^^^^^^^^^

PLUMED supports the :ref:`features <features-output>` output, and can use it as
collective variables to perform advanced sampling such as metadynamics.
Additionally, it also supports a custom output named ``"plumed::cv"``, with the
same semantics and metadata structure as the :ref:`features <features-output>`
output.

How to install the code
^^^^^^^^^^^^^^^^^^^^^^^

See the official `installation instruction`_ in PLUMED documentation.

How to use the code
^^^^^^^^^^^^^^^^^^^

See the official `syntax reference`_ in PLUMED documentation.

.. _installation instruction: https://www.plumed.org/doc-master/user-doc/html/_m_e_t_a_t_e_n_s_o_r_m_o_d.html
.. _syntax reference: https://www.plumed.org/doc-master/user-doc/html/_m_e_t_a_t_e_n_s_o_r.html
