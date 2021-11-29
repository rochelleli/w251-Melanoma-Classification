# w251-Melanoma-Classification

# Simulating Hand-held Melanoma Detection device

![Capture the Moles](https://github.com/rochelleli/w251-Melanoma-Classification/blob/main/iphone_capture.PNG)

Set up your Test directory with images you want to run predictions on
In our Simulation we selected 4 images 2 positive and 2 negative

## Run Docker our custom Docker container from your Jetson
sudo docker run -it --rm --runtime nvidia --network host -v /home/joslateriii/fp_data:/data -v /home/joslateriii/w251:/media joslateriii/melanoma-edge

## This will start Jupyter Notebook
Go to <IP_ADDRESS>:8888 in your browser

Download make sure the drives are mapped to where you have the GitHub repositoriy cloned

Open the Melanoma-Edge.pynb in Jupyter and run it

## Scan results
![Scan Results](https://github.com/rochelleli/w251-Melanoma-Classification/blob/main/scan_results.PNG)
