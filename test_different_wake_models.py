"""
Test different PyWake models to find one that computes wakes correctly.

The Blondel_Cathelain_2020 with Jimenez deflection isn't working.
Let's try simpler models.
"""

import numpy as np
from py_wake.site import UniformSite
from py_wake.examples.data.dtu10mw import DTU10MW
from py_wake import NOJ, BastankhahGaussian, IEA37SimpleBastankhahGaussian
from py_wake.deflection_models import JimenezWakeDeflection

D = 178.0
spacing = 5 * D

site = UniformSite()
wt = DTU10MW()

x2 = np.array([0.0, spacing])
y2 = np.array([0.0, 0.0])

models_to_test = [
    ("NOJ (Jensen)", NOJ(site, wt)),
    ("BastankhahGaussian", BastankhahGaussian(site, wt)),
    ("BastankhahGaussian + Jimenez", BastankhahGaussian(site, wt, deflectionModel=JimenezWakeDeflection())),
    ("IEA37 Bastankhah", IEA37SimpleBastankhahGaussian(site, wt)),
]

print("="*70)
print("Testing Different Wake Models")
print("="*70)
print(f"\nSetup: 2 turbines, {spacing/D}D = {spacing:.0f}m spacing")
print(f"Wind: 8 m/s from 0°")
print()

for name, model in models_to_test:
    print(f"\n{name}:")
    print("-" * 50)

    try:
        # Check if model needs tilt
        try:
            sim = model(x=x2, y=y2, wd=0, ws=8.0, yaw=0)
        except (ValueError, TypeError) as e:
            if 'tilt' in str(e):
                sim = model(x=x2, y=y2, wd=0, ws=8.0, yaw=0, tilt=0)
            else:
                raise

        P = sim.Power.values.flatten() / 1e6

        wake_loss_pct = (1 - P[1]/P[0]) * 100

        print(f"  T0: {P[0]:.3f} MW")
        print(f"  T1: {P[1]:.3f} MW")
        print(f"  Wake loss: {wake_loss_pct:.1f}%")

        if wake_loss_pct > 5:
            print(f"  ✓ Wakes ARE computed ({wake_loss_pct:.1f}% loss)")

            # Test wake steering
            print(f"\n  Testing wake steering with T0 yawed 25°:")
            try:
                yaw_array = np.array([25.0, 0.0]).reshape(2, 1, 1)
                try:
                    sim_yawed = model(x=x2, y=y2, wd=0, ws=8.0, yaw=yaw_array)
                except (ValueError, TypeError) as e:
                    if 'tilt' in str(e):
                        sim_yawed = model(x=x2, y=y2, wd=0, ws=8.0, yaw=yaw_array, tilt=0)
                    else:
                        raise

                P_yawed = sim_yawed.Power.values.flatten() / 1e6
                T0_loss = P_yawed[0] - P[0]
                T1_gain = P_yawed[1] - P[1]
                net = P_yawed.sum() - P.sum()

                print(f"    T0: {P_yawed[0]:.3f} MW ({T0_loss:+.3f})")
                print(f"    T1: {P_yawed[1]:.3f} MW ({T1_gain:+.3f})")
                print(f"    Net: {net:+.3f} MW")

                if T1_gain > 0.01:
                    print(f"    ✓ Wake steering WORKS! T1 gains {T1_gain:.3f} MW")
                    if net > 0:
                        print(f"    ✓✓ Net positive gain!")
                    else:
                        print(f"    ⚠ T1 gain doesn't compensate T0 loss")
                else:
                    print(f"    ❌ Wake steering doesn't help T1")

            except Exception as e:
                print(f"    Cannot test wake steering: {e}")

        else:
            print(f"  ❌ Wakes NOT computed (should be 30-50% loss)")

    except Exception as e:
        print(f"  ERROR: {e}")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

print("\nBased on results above:")
print("  1. Use a model that shows significant wake losses (30-50%)")
print("  2. Preferably one where wake steering provides T1 gains")
print("  3. Update nmpc_windfarm_acados_fixed.py to use that model")
