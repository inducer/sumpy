from pytools.tag import UniqueTag


class BaseKernelTag(UniqueTag):
    pass


# {{{ E2E kernel tags

class E2EBaseKernelTag(BaseKernelTag):
    pass


class E2EFromCSRKernelTag(E2EBaseKernelTag):
    pass


class E2EFromChildrenKernelTag(E2EBaseKernelTag):
    pass


class E2EFromParentKernelTag(E2EBaseKernelTag):
    pass

# }}}


# {{{ P2E kernel tags

class P2EBaseKernelTag(BaseKernelTag):
    pass


class P2EFromSingleBoxKernelTag(E2EBaseKernelTag):
    pass


class P2EFromCSRKernelTag(E2EBaseKernelTag):
    pass

# }}}


# {{{ P2P kernel tags

class P2PBaseKernelTag(BaseKernelTag):
    pass


class P2PKernelTag(P2PBaseKernelTag):
    pass


class P2PMatrixGeneratorKernelTag(P2PBaseKernelTag):
    pass


class P2PMatrixSubsetGeneratorKernelTag(P2PBaseKernelTag):
    pass


class P2PFromCSRKernelTag(P2PBaseKernelTag):
    pass

# }}}


# {{{ E2P kernel tags

class E2PBaseKernelTag(BaseKernelTag):
    pass


class E2PFromSingleBoxKernelTag(E2PBaseKernelTag):
    pass


class E2PFromCSRKernelTag(E2PBaseKernelTag):
    pass

# }}}
