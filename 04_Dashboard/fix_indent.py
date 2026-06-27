with open('app.py', 'r') as f:
    lines = f.readlines()

# Find and fix the indentation issue
fixed = []
in_variant_sim = False
for i, line in enumerate(lines):
    # Check if we're in the variant simulator section
    if 'def render_variant_simulator_page' in line:
        in_variant_sim = True
    # Find the line with the problematic code
    if 'input_vector = np.array' in line:
        # Skip this line
        continue
    if 'predicted_slope = float(model.predict(input_vector)' in line:
        # Skip this line
        continue
    if 'val_abs = abs(predicted_slope)' in line and in_variant_sim:
        # This is the line with indentation error - fix it
        fixed.append('        val_abs = abs(predicted_slope)\n')
        continue
    fixed.append(line)

with open('app.py', 'w') as f:
    f.writelines(fixed)

print("✅ Fixed indentation error")
