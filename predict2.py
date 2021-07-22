from glob import glob
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
import concurrent
from functools import partial
import sys
import torch
from torchvision import transforms


def predict(input_path, sample_path, model, device='cuda', max_col=10, max_row=13):
    model.eval()
    for sample_id in glob(f'{sample_path}/*.png'):
        result_img = np.zeros([256 * max_col, 256 * max_row, 3], dtype=np.float32)

        input_ids = [input_id for input_id in glob(f'{input_path}/*.png')
                              if sample_id.split('.')[0][-5:] in input_id]
        # print(input_ids)

        for i, input_id in tqdm(enumerate(input_ids)):
            col = i // max_row
            row = i % max_row

            # print(f'input: {input_id} col: {col} row: {row}')

            input_img = Image.open(input_id).convert('RGB')
            input_img = transforms.ToTensor()(input_img).unsqueeze(0).to(device)

            pred = model(input_img)
            pred = pred.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)

            result_img[col * 256:(col + 1) * 256, row * 256:(row + 1) * 256, :] = pred

        result_img = result_img[:2448, :3264, :]
        result_img = (result_img * 255.0).astype(np.uint8)
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(sample_id, result_img)
        # result_img = Image.fromarray(result_img).convert('RGB')
        # sys.exit()
        # result_img.save(sample_id)


def main():
    input_path = 'data/test_input_256_256'
    sample_path = 'data/sample_submission'

    model_path = "model0.pth"

    model = torch.hub.load(
            'mateuszbuda/brain-segmentation-pytorch', 'unet',
            in_channels=3, out_channels=3, init_features=32
        )
    model = model.to('cuda')
    model.load_state_dict(torch.load(model_path))

    predict(input_path, sample_path, model)

    # with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor: # windows local 환경시 max_workers=os.cpu_count()//2
    #     list(tqdm(
    #         executor.map(partial(predict, save_path='./data//'), train_all_input_files.to_list()),             
    #         desc='train image cut',
    #         total=len(train_all_input_files)
    #     ))


if __name__ == '__main__':
    main()