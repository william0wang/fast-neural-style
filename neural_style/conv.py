
import turicreate as tc

## Need pytorch 1.4 : pip install torch==1.4.0 torchvision==0.5.0
if __name__ == "__main__":

    # Load the style and content images
    styles = tc.load_images('style/')
    content = tc.load_images('content/')

    # Create a StyleTransfer model
    model = tc.style_transfer.create(styles, content)

    # Load some test images
    test_images = tc.load_images('test/')

    # Stylize the test images
    stylized_images = model.stylize(test_images)

    # Save the model for later use in Turi Create
    model.save('style-transfer.model')

    # Export for use in Core ML
    model.export_coreml('MyCustomStyleTransfer.mlmodel')
