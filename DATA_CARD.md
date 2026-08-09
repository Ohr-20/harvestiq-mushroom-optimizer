# Data card

Each row represents one crop production zone at a planning cutoff. The targets are forecast harvest kilograms and whether an operational stress event occurs during the crop-specific forecast window.

Supported systems are `grow_room`, `greenhouse`, `hydroponic`, and `orchard`. Supported crops are oyster mushroom, lion's mane, shiitake, avocado, tomato, strawberry, cucumber, lettuce, bell pepper, and basil.

Features cover crop, system, stage, zone, cycle age, production area, temperature, humidity, CO₂, root-zone or substrate moisture, ventilation, light exposure, previous yield, development index, and a derived crop-relative stress index.

The 9,000-row bundled dataset is synthetic and deterministic. It contains no personal or customer data. Crop profiles encode illustrative environmental centers and planning horizons; they are not field trials or proof of accuracy.

Real deployment requires consented farm records, calibrated sensors, stable collection times, crop/variety metadata, closed outcome windows, append-only raw storage, seasonal coverage, and performance reporting by crop and growing system.

