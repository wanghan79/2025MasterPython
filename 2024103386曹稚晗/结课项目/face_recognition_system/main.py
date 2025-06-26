import cv2
import argparse
import sys
from recognizer import FaceRecognizer
from utils.video_utils import VideoCapture
from utils.face_utils import load_image_file
from config.settings import Config


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Face Recognition System')

    # 视频源参数
    parser.add_argument('--source', type=str, default='0',
                        help='Video source (file path or camera index)')

    # 操作模式参数
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--add-face', action='store_true',
                       help='Add new face mode')
    group.add_argument('--list-faces', action='store_true',
                       help='List all known faces')
    group.add_argument('--clear-faces', action='store_true',
                       help='Clear all known faces')

    # 识别参数
    parser.add_argument('--name', type=str,
                        help='Name for the new face (required in add-face mode)')
    parser.add_argument('--align', action='store_true',
                        help='Align faces before recognition')
    parser.add_argument('--landmarks', action='store_true',
                        help='Show face landmarks')

    return parser.parse_args()


def add_face_mode(recognizer, args):
    """添加新人脸模式"""
    if not args.name:
        print("Error: --name is required in add-face mode")
        return False

    print(f"Adding new face: {args.name}. Press 's' to save, 'q' to quit.")
    cap = VideoCapture(args.source)

    try:
        for frame in cap:
            # 显示视频
            cv2.imshow("Add New Face", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):  # 保存
                rgb_frame = frame[:, :, ::-1]  # 转换为RGB
                recognizer.add_new_face(rgb_frame, args.name, args.align)
                print(f"Face for {args.name} added successfully!")
                return True
            elif key == ord('q'):  # 退出
                print("Operation cancelled.")
                return False
    finally:
        cap.release()
        cv2.destroyAllWindows()


def recognition_mode(recognizer, args):
    """人脸识别模式"""
    print("Starting face recognition. Press 'q' to quit.")
    cap = VideoCapture(args.source)

    try:
        for frame in cap:
            processed_frame, names = recognizer.process_frame(
                frame,
                align=args.align,
                landmarks=args.landmarks
            )
            cv2.imshow("Face Recognition", processed_frame)

            # 显示识别结果
            if names:
                print("Detected faces:", ", ".join(names))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    """主函数"""
    args = parse_arguments()
    Config.ensure_directories_exist()

    recognizer = FaceRecognizer()

    if args.add_face:
        add_face_mode(recognizer, args)
    elif args.list_faces:
        print("Known faces:")
        for name in recognizer.get_known_faces():
            print(f"- {name}")
    elif args.clear_faces:
        recognizer.clear_known_faces()
        print("All known faces have been cleared.")
    else:
        recognition_mode(recognizer, args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)