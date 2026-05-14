import numpy as np
import cdfmm

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
