from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="fGEsh6hVqfY168zf0CBs"
)

result = client.run_workflow(
    workspace_name="borosa",
    workflow_id="detect-count-and-visualize",
    images={
        "image": "./nulla.jpg"
    },
    use_cache=True 
)

# Your annotations mapping
annotations = {
    0: "crosswalk",
    1: "green",
    2: "no",
    3: "red"
}

# Your JSON data
data = result
# Extract and map class_ids
mapped_classes = [annotations[pred['class_id']] for pred in data[0]['predictions']['predictions']]

print(mapped_classes)
