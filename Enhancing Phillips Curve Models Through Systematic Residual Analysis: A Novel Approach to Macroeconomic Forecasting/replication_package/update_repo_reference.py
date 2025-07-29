#!/usr/bin/env python3
"""Update repository reference in paper.tex"""

# Read the LaTeX file
with open('outputs/phillips_curve_paper.tex', 'r') as f:
    content = f.read()

# Update the repository reference
old_text = "Complete code is available for replication, ensuring full reproducibility of results."
new_text = "Complete code is available for replication at \\url{https://github.com/VoxGenius/undismal-protocol/}, ensuring full reproducibility of results."

content = content.replace(old_text, new_text)

# Write back
with open('outputs/phillips_curve_paper.tex', 'w') as f:
    f.write(content)

print("✓ Updated repository reference to https://github.com/VoxGenius/undismal-protocol/")