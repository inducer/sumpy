
def test_m2l_creation(ctx, mpole_expn_class, local_expn_class, knl, order):
    from sympy.core.cache import clear_cache
    clear_cache()
    m_expn = mpole_expn_class(knl, order=order)
    l_expn = local_expn_class(knl, order=order)
    from sumpy.e2e import E2EFromCSR
    m2l = E2EFromCSR(ctx, m_expn, l_expn)
    import time
    start = time.time()
    m2l.run_translation_and_cse()
    return time.time() - start

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    from sumpy.kernel import HelmholtzKernel, LaplaceKernel

    import pyopencl as cl
    ctx = cl._csc()
    from sumpy.expansion.local import HelmholtzConformingVolumeTaylorLocalExpansion as LExpn
    from sumpy.expansion.multipole import HelmholtzConformingVolumeTaylorMultipoleExpansion as MExpn
    #from sumpy.expansion.local import H2DLocalExpansion as LExpn
    #from sumpy.expansion.multipole import H2DMultipoleExpansion as MExpn
    results = []
    for order in range(1, 2):
        results.append((order, test_m2l_creation(ctx, MExpn, LExpn, HelmholtzKernel(2, 'k'), order)))
    print("order\ttime (s)")
    for order, time in results:
        print("{}\t{:.2f}".format(order, time))
