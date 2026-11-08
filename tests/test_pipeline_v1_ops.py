import io
import unittest
import numpy as np
from PIL import Image

from photo_advisor.processor import process_image
from photo_advisor.types import Recipe
from photo_advisor.validator import validate_and_clamp_recipe
from photo_advisor.ops.tone import apply_shadows
from photo_advisor.ops.sharpen import apply_sharpen


def png(array, mode="RGB"):
    buf=io.BytesIO(); Image.fromarray(array,mode).save(buf,format="PNG"); return buf.getvalue()

def recipe(**kwargs):
    values=dict(pipeline_version="pipeline_v1",exposure=0.,contrast=1.,saturation=1.,crop=None,
        temperature=0.,tint=0.,shadows=0.,highlights=0.,sharpen=0.,straighten_deg=0.,local=None)
    values.update(kwargs); return Recipe(**values)

class PipelineV1OpsTests(unittest.TestCase):
    def test_neutral_recipe_is_pixel_identical(self):
        a=np.random.default_rng(3).integers(0,256,(48,51,3),dtype=np.uint8)
        result=process_image(png(a),recipe(),min_dimension_threshold=1)
        self.assertEqual(result.status,"success")
        np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(result.output_image_bytes))),a)

    def test_exposure_and_shadows_are_monotonic(self):
        a=np.full((24,24,3),64,dtype=np.uint8); data=png(a)
        low=process_image(data,recipe(),1); high=process_image(data,recipe(exposure=.5),1)
        self.assertGreaterEqual(np.asarray(Image.open(io.BytesIO(high.output_image_bytes))).mean(),np.asarray(Image.open(io.BytesIO(low.output_image_bytes))).mean())
        lin=np.full((3,3,3),.2,dtype=np.float32)
        self.assertGreaterEqual(apply_shadows(lin,.3).mean(),lin.mean())

    def test_output_finite_and_bounded_for_edges(self):
        cases=[np.zeros((4,4,3),np.uint8),np.full((4,4,3),255,np.uint8),np.array([[[90,120,180]]],np.uint8),np.full((3,4,4),100,np.uint8)]
        for arr in cases:
            mode="RGBA" if arr.shape[-1]==4 else "RGB"
            result=process_image(png(arr,mode),recipe(temperature=.5,shadows=.3,highlights=-.2,sharpen=.4),1)
            out=np.asarray(Image.open(io.BytesIO(result.output_image_bytes)))
            self.assertTrue(np.isfinite(out).all()); self.assertGreaterEqual(out.min(),0); self.assertLessEqual(out.max(),255)

    def test_deterministic_sha256(self):
        data=png(np.arange(32*32*3,dtype=np.uint8).reshape(32,32,3)); r=recipe(temperature=.2,shadows=.1,sharpen=.3)
        self.assertEqual(process_image(data,r,1).output_sha256,process_image(data,r,1).output_sha256)

    def test_sharpen_halo_bounded_on_step_edge(self):
        a=np.zeros((32,32,3),np.float32); a[:,16:]=1
        out=apply_sharpen(a,.5)
        self.assertGreaterEqual(float(out.min()),-.01); self.assertLessEqual(float(out.max()),1.01)

    def test_v1_validator_defaults_and_clamps(self):
        raw=dict(pipeline_version="pipeline_v1",exposure=0,contrast=1,saturation=1,crop=None,
            temperature=4,tint=-4,shadows=2,highlights=-2,sharpen=2,straighten_deg=8,local=None)
        v=validate_and_clamp_recipe(raw,800,800)
        self.assertTrue(v.is_valid,v.rejection_reason)
        self.assertEqual((v.effective_recipe.temperature,v.effective_recipe.tint,v.effective_recipe.shadows,
            v.effective_recipe.highlights,v.effective_recipe.sharpen,v.effective_recipe.straighten_deg),(1.,-1.,.5,-.5,1.,5.))
        self.assertEqual(len(v.warnings),6)

if __name__=="__main__": unittest.main()
