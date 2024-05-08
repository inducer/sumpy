import logging
import os

import numpy as np

import matplotlib.pyplot as plt

import pyopencl as cl

from boxtree import TreeBuilder
from boxtree.array_context import PyOpenCLArrayContext
from sumpy.array_context import PytatoPyOpenCLArrayContext

from boxtree.cost import FMMCostModel, make_pde_aware_translation_cost_model
from boxtree.fmm import drive_fmm
from boxtree.traversal import FMMTraversalBuilder
from boxtree.tools import make_normal_particle_array as p_normal

from functools import partial

from sumpy.expansion.multipole import VolumeTaylorMultipoleExpansion
from sumpy.expansion.local import VolumeTaylorLocalExpansion 
from sumpy.kernel import LaplaceKernel
from sumpy.expansion.m2l import NonFFTM2LTranslationClassFactory
from sumpy.fmm import (
    SumpyTreeIndependentDataForWrangler, 
    SumpyExpansionWrangler
)

logging.basicConfig(level=os.environ.get("LOGLEVEL", "WARNING"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def demo_cost_model(plot_results=False, lazy=False):

    # {{{ useful variables and actx setup

    nsources_list = [5000]
    ntargets_list = [5000]
    #nsources_list = [100, 200, 300, 400, 500]
    #ntargets_list = [100, 200, 300, 400, 500]
    nparticles_per_box_list = [32, 64, 128, 256, 512]
    #nparticles_per_box_list = [32, 64, 128]
    dim = 2
    dtype = np.float64

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    actx_boxtree = PyOpenCLArrayContext(queue, force_device_scalars=True)

    traversals = []
    traversals_dev = []
    level_orders_list = []
    timing_results = []
    results = {}
    fields = ["form_multipoles", "eval_direct", "multipole_to_local", 
              "eval_multipoles", "form_locals", "eval_locals", 
              "coarsen_multipoles", "refine_locals"]
    for field in fields:
        results[field] = []

    # }}}

    def fmm_level_to_order(kernel, kernel_args, tree, ilevel):
        return 10

    timings = {}
    for nparticles_per_box in nparticles_per_box_list:
        for nsources, ntargets in zip(nsources_list, ntargets_list):
            logger.info(f"Testing nsources = {nsources}, ntargets = {ntargets} "
                        f"with nparticles per box = {nparticles_per_box}")

            # {{{ Generate sources, targets and target_radii

            sources = p_normal(actx_boxtree, nsources, dim, dtype, seed=15)
            targets = p_normal(actx_boxtree, ntargets, dim, dtype, seed=18)

            rng = np.random.default_rng(seed=22)
            target_radii = rng.uniform(low=0.0, high=0.05, size=ntargets)

            # }}}

            # {{{ Generate tree and traversal

            tb = TreeBuilder(actx_boxtree)
            tree, _ = tb(
                actx_boxtree, sources, targets=targets, target_radii=target_radii,
                stick_out_factor=0.15, 
                max_particles_in_box=nparticles_per_box, debug=True
            )

            tg = FMMTraversalBuilder(actx_boxtree, well_sep_is_n_away=2)
            trav_dev, _ = tg(actx_boxtree, tree, debug=True)
            #trav = actx.to_numpy(trav_dev)
            trav = trav_dev

            traversals.append(trav)
            traversals_dev.append(trav_dev)

            # }}}
            
            # {{{ snag queue from eager tree building arraycontext

            if lazy:
                queue = actx_boxtree.queue
                actx = PytatoPyOpenCLArrayContext(queue)
            else:
                actx = actx_boxtree 

            # }}}

            # {{{ define kernel and expansion classes
            
            kernel = LaplaceKernel(dim)

            mpole_expansion_cls = VolumeTaylorMultipoleExpansion 
            local_expansion_cls = VolumeTaylorLocalExpansion
            m2l_factory = NonFFTM2LTranslationClassFactory()
            m2l = m2l_factory.get_m2l_translation_class(kernel, 
                                                        local_expansion_cls)()

            # }}}

            # {{{ define interface for fmm driver

            tree_indep = SumpyTreeIndependentDataForWrangler(
                actx,
                partial(mpole_expansion_cls, kernel),
                partial(local_expansion_cls, kernel, m2l_translation=m2l),
                [kernel])

            wrangler = SumpyExpansionWrangler(
                tree_indep, 
                trav,
                np.float64,
                fmm_level_to_order=fmm_level_to_order)

            level_orders_list.append(wrangler.level_orders)

            # }}}

            # {{{ fmm

            timing_data = {}
            src_weights = np.random.rand(tree.nsources).astype(tree.coord_dtype)
            drive_fmm(actx, wrangler, (src_weights,), timing_data=timing_data)
            timing_results.append(timing_data)

            # def driver(src_weights):
            #     return drive_fmm(actx, wrangler, (src_weights,),
            #                      timing_data=timing_data)
            #
            # src_weights = actx.from_numpy(src_weights)
            # actx.compile(driver)(src_weights)

            # }}}

        # {{{ build cost model 

        time_field_name = "process_elapsed"

        cost_model = FMMCostModel(make_pde_aware_translation_cost_model)

        model_results = []
        for icase in range(len(traversals)-1):
            traversal = traversals_dev[icase]
            model_results.append(
                cost_model.cost_per_stage(
                    actx_boxtree, traversal, level_orders_list[icase],
                    FMMCostModel.get_unit_calibration_params(),
                )
            )

        # }}}

        queue.finish()

        if not timing_results:
            return
        
        # {{{ analyze and report cost model results

        params = cost_model.estimate_calibration_params(
            model_results, timing_results[:-1], time_field_name=time_field_name
        )

        predicted_time = cost_model.cost_per_stage(
            actx_boxtree, traversals_dev[-1], level_orders_list[-1], params,
        )

        queue.finish()

        for field in ["form_multipoles", "eval_direct", "multipole_to_local",
                      "eval_multipoles", "form_locals", "eval_locals",
                      "coarsen_multipoles", "refine_locals"]:
            # measured = timing_results[-1][field]["process_elapsed"]
            # pred_err = (
            #         (measured - predicted_time[field])
            #         / measured)
            # logger.info("actual/predicted time for %s: %.3g/%.3g -> %g %% error",
            #         field,
            #         measured,
            #         predicted_time[field],
            #         abs(100*pred_err))

            if nparticles_per_box != nparticles_per_box_list[0]:
                results[field].append(predicted_time[field])

        # }}}


    if plot_results:
        x = np.arange(len(nparticles_per_box_list) - 1)
        width = 0.1
        mult = 0

        fig, ax = plt.subplots()

        for field, timing in results.items():
            offset = width*mult
            bars = ax.bar(x + offset, timing, width, label=field)
            #ax.bar_label(bars, padding=4)
            mult += 1

        ax.set_xlabel("Particles per box")
        ax.set_xticks(x + 3*width, nparticles_per_box_list[1:])

        ax.set_ylabel("Process time (s)")

        ax.legend(loc="upper left", ncols=3)

        #plt.show()
        plt.savefig("./balancing.pdf")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--lazy", action="store_true")
    parser.add_argument("--plot_results", action="store_true")

    args = parser.parse_args()

    demo_cost_model(lazy=args.lazy, plot_results=args.plot_results)
