# Grid refinement (under development)

Grid refinement (patch and conservative AMR) is under development and not
validated; it is not part of the supported feature set.

This page is not listed in the navigation. The earlier showcase results were
withdrawn: they did not demonstrate a cost or accuracy gain on Nu or Cd, and the reasons are
recorded in [Limitations](../theory/10_limitations.md):

- Patch-refined thermal walls are not half-way: in pure conduction the
  patch-refined runs give Nu of about 1.3 instead of 1.0 (issue #98).
- The conservative-tree AMR drag coefficient is off by a factor between 1.06
  and 5.6 against the reference.
