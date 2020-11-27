from preprocessing import preprocessing
from FirstDataset import FirstDataset

if __name__ == '__main__':
    # preprocessing( 'D:\\2020\\PCO项目\\严院长病人前节PCO\\可用图像')
    dataset = FirstDataset(img_dir='D:\\2020\\PCO项目\\严院长病人前节PCO\\可用图像',
                           annotation_dir='D:\\2020\\PCO项目\\严院长病人前节PCO\\label')

    pass
    # Get raw slit-lamp image dataset

    # Data preprocessing

    # Train self-supervised representaion learning model

    # Fine-tune with ucva values

    # Clustering

    # Evaluate
