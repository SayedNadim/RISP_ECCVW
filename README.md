# RISP_ECCVW_2022
The official code for our RGB-to-RAW model in "Reverse ISP Challenge - Track 1 - S7".
## To train the model:
- Python 3.8/PyTorch 1.8
- Install kornia (`pip install kornia`), pyyaml (`pip install pyyaml`) and tqdm (`pip install tqdm`) and . I assume you have rawpy installed - optional. If code throws error for rawpy, comment out `import rawpy` in `data_pipeline/data_utils`. 
- Modify the 'train.config' file.
- Train the model.

## To test the model
- Modify the 'test.config' file.
- Load the pre-trained model (args.resume) in the test script and run
