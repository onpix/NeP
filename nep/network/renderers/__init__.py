from nep.network.renderers.shape import NePShapeRenderer
from nep.network.renderers.nerf import NeRFRenderer
from nep.network.renderers.material import NePMaterialRenderer
from nep.network.renderers.neilf import NeILFModel

name2renderer = {
    "nerf": NeRFRenderer,
    "shape": NePShapeRenderer,
    "material": NePMaterialRenderer,
    "neilf": NeILFModel,
}
