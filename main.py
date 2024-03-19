import argparse
import copy
import csv
import time

import cv2
import numpy as np
import tqdm
from PIL import Image
from face_detection import RetinaFace
from torch.utils.data import DataLoader

from nets import nn
from utils.datasets import Datasets
from utils.util import *


def train(args):
    weight_path = f'outputs/weights/{args.model_name}.pth.tar'
    model = nn.HPE(weight_path, True, args.model_name).cuda()
    dataset = Datasets(f'{args.data_dir}', 'AFLW2K', get_transforms(True))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    criterion = GeodesicLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.5)
    ema = nn.EMA(model)

    best = float('inf')
    with open('outputs/weights/log.csv', 'w') as log:
        logger = csv.DictWriter(log, fieldnames=['epoch', 'Pitch', 'Yaw', 'Roll'])
        logger.writeheader()
        for epoch in range(args.epochs):
            print(('\n' + '%10s' * 3) % ('epoch', 'memory', 'loss'))
            p_bar = tqdm.tqdm(loader, total=len(loader))
            model.train()
            total_loss = 0
            for i, (samples, labels) in enumerate(p_bar):
                samples = samples.cuda()
                labels = labels.cuda()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                outputs = model(samples)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                memory = f'{torch.cuda.memory_reserved() / 1E9:.3g}G'
                s = ('%10s' * 2 + '%10.3g') % (f'{epoch + 1}/{args.epochs}', memory, loss.item())
                p_bar.set_description(s)

            last = test(args, ema.ema)

            logger.writerow({'Pitch': str(f'{last[0]:.3f}'),
                             'Yaw': str(f'{last[1]:.3f}'),
                             'Roll': str(f'{last[2]:.3f}'),
                             'epoch': str(epoch + 1).zfill(3)})
            log.flush()
            if sum(last) < best:
                best = sum(last)
                print('....saved best')
                save = {'model': copy.deepcopy(ema.ema).half()}
                torch.save(save, f'outputs/weights/best.pt')
                del save

            save = {'model': copy.deepcopy(ema.ema).half()}
            torch.save(save, f'outputs/weights/last.pt')
            del save

        scheduler.step()

    torch.cuda.empty_cache()
    print('Training completed.')


@torch.no_grad()
def test(args, model=None):
    if model is None:
        model = torch.load(f=f'outputs/weights/best.pt', map_location='cuda')['model'].float()
    model = nn.re_parameterize_model(model)
    model.eval()
    model.inference_mode = True

    dataset = Datasets(f'{args.data_dir}', 'AFLW2K', get_transforms(False), False)
    loader = DataLoader(dataset, batch_size=64)

    total, y_error, p_error, r_error = 0, 0.0, 0.0, 0.0
    for sample, label in tqdm.tqdm(loader, ('%10s' * 3) % ('Pitch', 'Yaw', 'Roll')):
        sample = sample.cuda()
        total += label.size(0)

        p_gt = label[:, 0].float() * 180 / np.pi
        y_gt = label[:, 1].float() * 180 / np.pi
        r_gt = label[:, 2].float() * 180 / np.pi

        output = model(sample)
        euler = compute_euler(output) * 180 / np.pi

        p_pred = euler[:, 0].cpu()
        y_pred = euler[:, 1].cpu()
        r_pred = euler[:, 2].cpu()

        p_error += torch.sum(torch.min(torch.stack((torch.abs(p_gt - p_pred),
                                                    torch.abs(p_pred + 360 - p_gt),
                                                    torch.abs(p_pred - 360 - p_gt),
                                                    torch.abs(p_pred + 180 - p_gt),
                                                    torch.abs(p_pred - 180 - p_gt))), 0)[0])

        y_error += torch.sum(torch.min(torch.stack((torch.abs(y_gt - y_pred),
                                                    torch.abs(y_pred + 360 - y_gt),
                                                    torch.abs(y_pred - 360 - y_gt),
                                                    torch.abs(y_pred + 180 - y_gt),
                                                    torch.abs(y_pred - 180 - y_gt))), 0)[0])

        r_error += torch.sum(torch.min(torch.stack((torch.abs(r_gt - r_pred),
                                                    torch.abs(r_pred + 360 - r_gt),
                                                    torch.abs(r_pred - 360 - r_gt),
                                                    torch.abs(r_pred + 180 - r_gt),
                                                    torch.abs(r_pred - 180 - r_gt))), 0)[0])

    p_error, y_error, r_error = p_error / total, y_error / total, r_error / total
    # avg_error = (p_error + y_error + r_error) / (3 * total)
    print(('%10.3g' * 3) % (p_error, y_error, r_error))

    model.float()  # for training
    return p_error, y_error, r_error


@torch.no_grad()
def inference(args):
    model = nn.HPE(f'{args.save_dir}/weights/last.pt', False, args.model_name)['model']
    model = nn.re_parameterize_model(model)
    model.cpu().eval()
    model.inference_mode = True

    detector = RetinaFace(0)

    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(f'{args.save_dir}/output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25,
                          (frame_width, frame_height))
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    with torch.no_grad():
        while True:
            ret, frame = cap.read()

            faces = detector(frame)

            for box, landmarks, score in faces:
                if score < .95:
                    continue
                x_min = int(box[0])
                y_min = int(box[1])
                x_max = int(box[2])
                y_max = int(box[3])
                bbox_width = abs(x_max - x_min)
                bbox_height = abs(y_max - y_min)

                x_min = max(0, x_min - int(0.2 * bbox_height))
                y_min = max(0, y_min - int(0.2 * bbox_width))
                x_max = x_max + int(0.2 * bbox_height)
                y_max = y_max + int(0.2 * bbox_width)

                img = frame[y_min:y_max, x_min:x_max]
                img = Image.fromarray(img)
                img = img.convert('RGB')
                img = get_transforms(False)(img)

                img = torch.Tensor(img[None, :]).cpu()

                c = cv2.waitKey(1)
                if c == 27:
                    break

                start = time.time()
                R_pred = model(img)
                end = time.time()
                print('Head pose estimation: %2f ms' % ((end - start) * 1000.))

                euler = compute_euler(
                    R_pred) * 180 / np.pi
                p_pred_deg = euler[:, 0].cpu()
                y_pred_deg = euler[:, 1].cpu()
                r_pred_deg = euler[:, 2].cpu()

                # utils.draw_axis(frame, y_pred_deg, p_pred_deg, r_pred_deg, left+int(.5*(right-left)), top, size=100)
                plot_pose_cube(frame, y_pred_deg, p_pred_deg, r_pred_deg, x_min + int(.5 * (
                        x_max - x_min)), y_min + int(.5 * (y_max - y_min)), size=bbox_width)

            cv2.imshow("Demo", frame)
            out.write(frame)
            cv2.waitKey(5)
        cap.release()
        out.release()

        # Closes all the frames
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Head Pose Estimation')
    parser.add_argument('--model_name', type=str, default='s1')
    parser.add_argument('--data_dir', type=str, default='../../Datasets/HPE')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', default=True, action='store_true')
    parser.add_argument('--inference', action='store_true')

    args = parser.parse_args()
    if args.train:
        train(args)
    if args.test:
        test(args)
    if args.inference:
        inference(args)


if __name__ == "__main__":
    main()
