import os
import argparse
from pydub import AudioSegment
from pydub.utils import make_chunks

def main(args):
    def chunk_and_save(file):
        try:
            audio = AudioSegment.from_file(file)
            length = args.seconds * 1000  # this is in milliseconds
            chunks = make_chunks(audio, length)
            names = []
            for i, chunk in enumerate(chunks):
                _name = os.path.basename(file)
                name = "{}_{}".format(i, _name)
                wav_path = os.path.join(args.save_path, name)
                chunk.export(wav_path, format="wav")
                names.append(wav_path)
            print(f"Successfully processed: {file} into {len(chunks)} chunks")
            return names
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            return []

    # Create save directory if it doesn't exist
    os.makedirs(args.save_path, exist_ok=True)
    
    # Check if input is a directory
    if os.path.isdir(args.audio_dir):
        all_names = []
        processed_count = 0
        
        # Function to process audio files in a directory and its subdirectories
        def process_directory(directory):
            nonlocal all_names, processed_count
            
            # Get all items in the directory
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                
                # If it's a directory, recursively process it
                if os.path.isdir(item_path):
                    process_directory(item_path)
                # If it's an audio file, process it
                elif item.lower().endswith(('.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac')):
                    names = chunk_and_save(item_path)
                    all_names.extend(names)
                    processed_count += 1
        
        # Start recursive processing from the root directory
        process_directory(args.audio_dir)
        print(f"Processed {processed_count} audio files into {len(all_names)} chunks")
    else:
        print("Error: Input directory does not exist")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="script to split audio files in a directory (and subdirectories) into chunks")
    parser.add_argument('--seconds', type=int, default=10,
                        help='length of each chunk in seconds')
    parser.add_argument('--audio_dir', type=str, default=None, required=True,
                        help='directory containing audio files to process (will search recursively)')
    parser.add_argument('--save_path', type=str, default=None, required=True,
                        help='full path to save chunked files. i.e. /to/path/saved_clips/')
    parser.add_argument('--preserve_structure', action='store_true',
                        help='preserve the directory structure in the output')

    args = parser.parse_args()

    main(args)