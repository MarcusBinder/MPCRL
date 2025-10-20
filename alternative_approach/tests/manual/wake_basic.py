"""
Basic test: Is PyWake computing wakes at all?

T1 should have LESS power than T0 if it's in the wake.
"""

import numpy as np
from py_wake.site import UniformSite
from py_wake.examples.data.dtu10mw import DTU10MW
from py_wake.literature.gaussian_models import Blondel_Cathelain_2020
from py_wake.turbulence_models import CrespoHernandez
from py_wake.deflection_models.jimenez import JimenezWakeDeflection

D = 178.0
spacing = 5 * D

site = UniformSite()
wt = DTU10MW()

wf_model = Blondel_Cathelain_2020(
    site, wt,
    turbulenceModel=CrespoHernandez(),
    deflectionModel=JimenezWakeDeflection()
)

print("="*70)
print("Basic Wake Test")
print("="*70)

# Test 1: Single turbine (no wake)
print("\n1. Single turbine (no wake effects):")
x1 = np.array([0.0])
y1 = np.array([0.0])
sim1 = wf_model(x=x1, y=y1, wd=0, ws=8.0, yaw=0, tilt=0)
P1 = sim1.Power.values.flatten()[0] / 1e6
print(f"   Power: {P1:.3f} MW")
print(f"   (This is the 'free stream' power)")

# Test 2: Two turbines, far apart (no interaction)
print("\n2. Two turbines, 100D apart (no wake interaction):")
x_far = np.array([0.0, 100*D])
y_far = np.array([0.0, 0.0])
sim_far = wf_model(x=x_far, y=y_far, wd=0, ws=8.0, yaw=0, tilt=0)
P_far = sim_far.Power.values.flatten() / 1e6
print(f"   T0: {P_far[0]:.3f} MW")
print(f"   T1: {P_far[1]:.3f} MW")
print(f"   → Both should be ~{P1:.3f} MW (no wake effect)")

# Test 3: Two turbines at 5D (IN WAKE)
print("\n3. Two turbines at 5D spacing (T1 IN WAKE):")
x2 = np.array([0.0, spacing])
y2 = np.array([0.0, 0.0])
sim2 = wf_model(x=x2, y=y2, wd=0, ws=8.0, yaw=0, tilt=0)
P2 = sim2.Power.values.flatten() / 1e6
print(f"   T0: {P2[0]:.3f} MW")
print(f"   T1: {P2[1]:.3f} MW")

wake_loss_pct = (1 - P2[1]/P2[0]) * 100
print(f"   → T1 wake loss: {wake_loss_pct:.1f}%")

if P2[1] < P2[0] * 0.95:  # More than 5% loss
    print(f"   ✓ Wake effects ARE working (T1 has {wake_loss_pct:.1f}% less power)")
else:
    print(f"   ❌ Wake effects NOT working (T1 should have significantly less power!)")
    print(f"\n   PROBLEM: PyWake is not computing wakes correctly!")
    print(f"   Possible causes:")
    print(f"     • Wrong wake model configuration")
    print(f"     • Wind direction issue")
    print(f"     • Turbine positions issue")

# Test 4: Check wind direction
print("\n4. Check effect of wind direction:")
print("   If wind blows from T0 → T1, T1 should be in wake")
print("   If wind blows perpendicular, no wake")

# Wind from west (0°) - T1 downstream
sim_0 = wf_model(x=x2, y=y2, wd=0, ws=8.0, yaw=0, tilt=0)
P_0 = sim_0.Power.values.flatten() / 1e6

# Wind from north (90°) - T1 not downstream
sim_90 = wf_model(x=x2, y=y2, wd=90, ws=8.0, yaw=0, tilt=0)
P_90 = sim_90.Power.values.flatten() / 1e6

print(f"\n   Wind from 0° (T0 → T1):")
print(f"     T0: {P_0[0]:.3f} MW, T1: {P_0[1]:.3f} MW")
print(f"\n   Wind from 90° (perpendicular):")
print(f"     T0: {P_90[0]:.3f} MW, T1: {P_90[1]:.3f} MW")

if abs(P_90[1] - P_90[0]) < 0.1 and P_0[1] < P_0[0]:
    print(f"\n   ✓ Wind direction works correctly")
elif P_0[1] >= P_0[0] * 0.95:
    print(f"\n   ❌ T1 not in wake even with wd=0° - Something is wrong!")

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

if P2[1] >= P2[0] * 0.95:
    print("\n❌ PROBLEM FOUND: PyWake is NOT computing wakes!")
    print("\nThis explains everything:")
    print("  • If turbines don't experience wake losses...")
    print("  • Then yawing provides NO benefit (only cosine loss)")
    print("  • This is why ψ=[0,0,0,0] appears optimal")
    print("\nPossible fixes:")
    print("  1. Check PyWake version and model compatibility")
    print("  2. Verify turbine power curve is loaded")
    print("  3. Try different wake model")
    print("  4. Check if specific model parameters needed")
else:
    print(f"\n✓ PyWake IS computing wakes ({wake_loss_pct:.1f}% loss)")
    print("\nBut wake steering still doesn't help because:")
    print("  • Straight-line layout")
    print("  • Deflected wake still hits downstream turbines")
    print("  • Need cross-wind separation for benefits")
