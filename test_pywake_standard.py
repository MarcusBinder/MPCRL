"""
Test wake steering with standard PyWake interface.

Verify that:
1. PyWake wake steering actually works
2. Our code is calling PyWake correctly
3. Understand what the expected behavior should be
"""

import numpy as np
import matplotlib.pyplot as plt
from py_wake.site import UniformSite
from py_wake.examples.data.dtu10mw import DTU10MW
from py_wake.literature.gaussian_models import Blondel_Cathelain_2020
from py_wake.turbulence_models import CrespoHernandez
from py_wake.deflection_models.jimenez import JimenezWakeDeflection

# Setup
D = 178.0
spacing = 5 * D

# Create site and turbine
site = UniformSite()
wt = DTU10MW()

# Create wind farm model with wake steering
wf_model = Blondel_Cathelain_2020(
    site, wt,
    turbulenceModel=CrespoHernandez(),
    deflectionModel=JimenezWakeDeflection()
)

print("="*70)
print("PyWake Wake Steering Test")
print("="*70)

# Test 1: 2 turbines in a row
print("\n1. Two turbines in a row (5D spacing)")
print("-" * 70)

x = np.array([0.0, spacing])
y = np.array([0.0, 0.0])

# Case A: Both aligned
sim_aligned = wf_model(x=x, y=y, wd=0, ws=8.0, yaw=0, tilt=0)
P_aligned = sim_aligned.Power.values.flatten() / 1e6

print(f"Aligned (ψ=[0, 0]):")
print(f"  T0: {P_aligned[0]:.3f} MW")
print(f"  T1: {P_aligned[1]:.3f} MW (in full wake)")
print(f"  Total: {P_aligned.sum():.3f} MW")

# Case B: T0 yawed +25°
yaw_steering = np.array([25.0, 0.0]).reshape(2, 1, 1)
sim_yawed = wf_model(x=x, y=y, wd=0, ws=8.0, yaw=yaw_steering, tilt=0)
P_yawed = sim_yawed.Power.values.flatten() / 1e6

print(f"\nT0 yawed 25° (ψ=[25, 0]):")
print(f"  T0: {P_yawed[0]:.3f} MW (cosine loss)")
print(f"  T1: {P_yawed[1]:.3f} MW")
print(f"  Total: {P_yawed.sum():.3f} MW")

T0_loss = P_yawed[0] - P_aligned[0]
T1_gain = P_yawed[1] - P_aligned[1]
net_gain = P_yawed.sum() - P_aligned.sum()

print(f"\nChanges:")
print(f"  T0 loss: {T0_loss:+.3f} MW")
print(f"  T1 gain: {T1_gain:+.3f} MW")
print(f"  Net:     {net_gain:+.3f} MW")

if T1_gain > 0.01:
    print("  ✓ Wake steering IS working!")
else:
    print("  ❌ Wake steering NOT working (T1 unchanged)")

# Test 2: 4 turbines in a row
print("\n" + "="*70)
print("2. Four turbines in a row (5D spacing)")
print("-" * 70)

x4 = np.array([0.0, spacing, 2*spacing, 3*spacing])
y4 = np.array([0.0, 0.0, 0.0, 0.0])

# Case A: All aligned
sim4_aligned = wf_model(x=x4, y=y4, wd=0, ws=8.0, yaw=0, tilt=0)
P4_aligned = sim4_aligned.Power.values.flatten() / 1e6

print(f"All aligned (ψ=[0, 0, 0, 0]):")
for i, p in enumerate(P4_aligned):
    print(f"  T{i}: {p:.3f} MW")
print(f"  Total: {P4_aligned.sum():.3f} MW")

# Case B: Upstream yawed (classic wake steering strategy)
yaw4_steering = np.array([25.0, 25.0, 25.0, 0.0]).reshape(4, 1, 1)
sim4_yawed = wf_model(x=x4, y=y4, wd=0, ws=8.0, yaw=yaw4_steering, tilt=0)
P4_yawed = sim4_yawed.Power.values.flatten() / 1e6

print(f"\nUpstream yawed (ψ=[25, 25, 25, 0]):")
for i, p in enumerate(P4_yawed):
    change = p - P4_aligned[i]
    print(f"  T{i}: {p:.3f} MW ({change:+.3f})")
print(f"  Total: {P4_yawed.sum():.3f} MW ({P4_yawed.sum() - P4_aligned.sum():+.3f})")

if P4_yawed.sum() > P4_aligned.sum():
    print("\n  ✓ Wake steering provides net benefit!")
else:
    print("\n  ❌ Wake steering DECREASES total power")

# Test 3: What if we use different deflection model or parameters?
print("\n" + "="*70)
print("3. Checking wake deflection model")
print("-" * 70)

# Check if deflection model is actually being used
print(f"Wake model: {wf_model.__class__.__name__}")
print(f"Deflection model: {wf_model.deflectionModel.__class__.__name__}")
print(f"Turbulence model: {wf_model.turbulenceModel.__class__.__name__}")

# Test 4: Plot flow field to see if wake is actually deflected
print("\n" + "="*70)
print("4. Visualizing wake deflection")
print("-" * 70)

# Create 2-turbine case for visualization
x2 = np.array([0.0, spacing])
y2 = np.array([0.0, 0.0])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Aligned case
sim = wf_model(x=x2, y=y2, wd=0, ws=8.0, yaw=0, tilt=0)
flow_map_aligned = sim.flow_map(wd=0, ws=8.0, yaw=0, tilt=0)
flow_map_aligned.plot_wake_map(ax=ax1, levels=20)
ax1.set_title("Aligned: ψ=[0°, 0°]")
ax1.plot(x2, y2, 'r^', markersize=10, label='Turbines')
ax1.legend()

# Yawed case
yaw2 = np.array([25.0, 0.0]).reshape(2, 1, 1)
sim = wf_model(x=x2, y=y2, wd=0, ws=8.0, yaw=yaw2, tilt=0)
flow_map_yawed = sim.flow_map(wd=0, ws=8.0, yaw=yaw2, tilt=0)
flow_map_yawed.plot_wake_map(ax=ax2, levels=20)
ax2.set_title("T0 Yawed: ψ=[25°, 0°]")
ax2.plot(x2, y2, 'r^', markersize=10, label='Turbines')
ax2.legend()

plt.tight_layout()
plt.savefig('wake_deflection_test.png', dpi=150)
print("Saved visualization: wake_deflection_test.png")
print("\nIf wake deflection works, you should see:")
print("  • Left: Wake centered on T1 (red=low wind speed)")
print("  • Right: Wake deflected AWAY from T1 (less red at T1 location)")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

if T1_gain > 0.01:
    print("\n✓ PyWake wake steering IS working")
    print("  → The problem is NOT with PyWake")
    print("  → Need to investigate MPC formulation")
else:
    print("\n❌ PyWake wake steering is NOT providing benefits")
    print("\nPossible reasons:")
    print("  1. Jimenez deflection model has limitations")
    print("  2. Straight-line layout genuinely gets no benefit")
    print("  3. Model parameters need tuning")
    print("  4. Need different wake/deflection model")
    print("\nRecommendation:")
    print("  • Check PyWake documentation for expected wake steering gains")
    print("  • Try different deflection models (e.g., Bastankhah)")
    print("  • Verify this is physically realistic for this turbine/spacing")
