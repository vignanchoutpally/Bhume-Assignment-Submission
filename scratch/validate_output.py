import geopandas as gpd

path = "Malatavadi_predictions.geojson"
gdf = gpd.read_file(path)

print(f"File loaded successfully: {path}")
print(f"Number of features: {len(gdf)}")
print("Columns:", list(gdf.columns))

# Verify columns
required = {'plot_number', 'status', 'confidence', 'geometry'}
missing = required - set(gdf.columns)
if missing:
    print(f"FAIL: Missing required columns: {missing}")
else:
    print("PASS: All required columns present.")

# Check status values
statuses = set(gdf['status'].unique())
print("Unique status values:", statuses)
invalid_statuses = statuses - {'corrected', 'flagged', 'CORRECTED', 'FLAGGED'}
if invalid_statuses:
    print(f"FAIL: Invalid status values found: {invalid_statuses}")
else:
    print("PASS: Status values are valid.")

# Check confidence values
conf_min = gdf['confidence'].min()
conf_max = gdf['confidence'].max()
print(f"Confidence range: [{conf_min:.4f}, {conf_max:.4f}]")
if conf_min < 0.0 or conf_max > 1.0:
    print("FAIL: Confidence values outside [0, 1]")
else:
    print("PASS: Confidence values in range.")

# Check unique plot numbers
num_unique = gdf['plot_number'].nunique()
print(f"Unique plot numbers: {num_unique} out of {len(gdf)}")
if num_unique != len(gdf):
    print("FAIL: Plot numbers are not unique!")
else:
    print("PASS: Plot numbers are unique.")

# Print sample
print("\nSample predictions:")
print(gdf[['plot_number', 'status', 'confidence']].head(10))
