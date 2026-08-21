import numpy as np
import cdfmm


def test_dense_direct_static_precision_and_input_dtypes():
    sources32 = np.array([[0.0, 0.0, 0.0], [1.0, -0.5, 0.25]], dtype=np.float32)
    targets64 = np.array([[0.0, 0.0, 2.0], [2.0, 1.0, 3.0]], dtype=np.float64)
    moments32 = np.array([[1.0, 2.0, 3.0], [-2.0, 1.0, 0.5]], dtype=np.float32)
    moments64 = moments32.astype(np.float64)

    fp32 = cdfmm.DenseDirectPlan(sources32, targets64)
    fp64 = cdfmm.DenseDirectPlan(
        sources32.astype(np.float64), targets64, static_precision="float64"
    )

    assert fp32.static_precision == cdfmm.StaticPrecision.FLOAT32
    assert fp32.tensor_memory_bytes * 2 == fp64.tensor_memory_bytes
    np.testing.assert_allclose(fp32.evaluate(moments32), fp64.evaluate(moments64),
                               rtol=2.0e-6, atol=1.0e-8)
    np.testing.assert_allclose(fp32.evaluate(moments64), fp32.evaluate(moments32),
                               rtol=0.0, atol=0.0)

def test_import():
    assert cdfmm is not None

def test_axial():
    r=cdfmm.p2p_dipole_pair([1,0,0],[0,0,0],[1,0,0],output="both")
    c=1/(4*np.pi)
    assert np.isclose(r["phi"],c)
    assert np.isclose(r["H"][0],2*c)

def test_transverse():
    r=cdfmm.p2p_dipole_pair([0,1,0],[0,0,0],[1,0,0],output="both")
    c=1/(4*np.pi)
    assert np.isclose(r["phi"],0.0)
    assert np.isclose(r["H"][0],-c)

def test_output_modes():
    rf=cdfmm.p2p_dipole_pair([1,0,0],[0,0,0],[1,0,0],output="field")
    rp=cdfmm.p2p_dipole_pair([1,0,0],[0,0,0],[1,0,0],output="potential")
    rb=cdfmm.p2p_dipole_pair([1,0,0],[0,0,0],[1,0,0],output="both")
    assert rf["phi"]==0.0
    assert np.allclose(rp["H"],0.0)
    assert rb["phi"]!=0.0 and not np.allclose(rb["H"],0.0)
