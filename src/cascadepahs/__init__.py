"""CascadePAHs – robotic sensing–sampling and privileged-information learning for
high-frequency surrogate reconstruction of dissolved ΣPAHs.

Main entry points
-----------------
* :class:`cascadepahs.models.SOLACE` – Sensor-to-Optical Leaf-Mapping Cascaded Estimator
* :func:`cascadepahs.models.SORFE`   – Sensor-Only Random-Forest Estimator (baseline)
* :mod:`cascadepahs.validation`      – date-grouped validation schemes
* :mod:`cascadepahs.calibration`     – few-shot cross-site calibration
* :mod:`cascadepahs.optics`          – UV–vis / EEM optical descriptors
* :mod:`cascadepahs.parafac`         – non-negative PARAFAC for EEM cubes
"""
__version__ = "1.0.0"
