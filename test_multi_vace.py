#!/usr/bin/env python3
"""
Test Multi-VACE Task Creation
============================

This script demonstrates the difference between the standalone example_multi_vace_python.py
and the proper integration with the headless database system.
"""

import sys
from pathlib import Path

def test_standalone_script():
    """Run the standalone example script"""
    print("🔍 Testing standalone example_multi_vace_python.py:")
    print("=" * 60)
    
    try:
        # Import and run the standalone script
        from example_multi_vace_python import main as example_main
        example_main()
        
        print("\n✅ Standalone script completed successfully!")
        print("📝 Note: This creates task parameters but doesn't add them to the database.")
        print("   The headless system won't see these tasks.")
        
    except Exception as e:
        print(f"❌ Standalone script failed: {e}")

def test_database_integration():
    """Test the database integration"""
    print("\n🔗 Testing database integration with create_multi_vace_task.py:")
    print("=" * 60)
    
    try:
        # Check if we have the video file
        video_path = "input.mp4"
        if not Path(video_path).exists():
            print(f"⚠️  Video file not found: {video_path}")
            print("   Creating a dummy video for testing...")
            
            # Create a simple test video using opencv
            try:
                import cv2
                import numpy as np
                
                # Create a 5-second test video
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = 24
                frames = fps * 5  # 5 seconds
                width, height = 640, 480
                
                out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                
                for i in range(frames):
                    # Create a simple gradient frame with changing colors
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
                    color_shift = int(255 * i / frames)
                    frame[:, :] = [color_shift, 128, 255 - color_shift]
                    
                    # Add some text
                    cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    out.write(frame)
                
                out.release()
                print(f"✅ Created test video: {video_path}")
                
            except ImportError:
                print("❌ OpenCV not available, cannot create test video")
                return
        
        # Now test the database integration
        from create_multi_vace_task import create_multi_vace_database_task
        
        task_id = create_multi_vace_database_task(
            guidance_video_path=video_path,
            output_dir="./test_output",
            max_frames_to_process=20,  # Smaller for testing
            context_frames_str="0:8,15,19",
            reference_frame_indices="0, 19",
            prompt="A simple test video with changing colors"
        )
        
        if task_id:
            print(f"\n✅ Database integration test successful!")
            print(f"   Task ID: {task_id}")
            print(f"   Status: Added to database")
            
            # Show how to check the database
            print(f"\n📋 To check the task in the database:")
            print(f"   python check_tasks.py")
            print(f"\n🎬 To process the task:")
            print(f"   python headless.py")
        else:
            print(f"❌ Database integration test failed")
            
    except Exception as e:
        print(f"❌ Database integration test failed: {e}")
        import traceback
        traceback.print_exc()

def show_comparison():
    """Show the key differences"""
    print("\n📊 KEY DIFFERENCES:")
    print("=" * 60)
    print("❌ example_multi_vace_python.py (standalone):")
    print("   • Creates task parameters in memory")
    print("   • Saves reference frames to disk")
    print("   • Prints parameters to console")
    print("   • Does NOT add anything to the database")
    print("   • Headless system never sees these tasks")
    
    print("\n✅ create_multi_vace_task.py (integrated):")
    print("   • Creates task parameters in memory")
    print("   • Saves reference frames AND guidance video to disk") 
    print("   • Converts PIL images to file paths")
    print("   • Adds complete task to SQLite database")
    print("   • Headless system picks up and processes these tasks")
    
    print("\n🔄 WORKFLOW:")
    print("   1. Run: python create_multi_vace_task.py")
    print("   2. Check: python check_tasks.py") 
    print("   3. Process: python headless.py")
    print("   4. Monitor: Check outputs/ directory")

def main():
    """Main test function"""
    print("Multi-VACE Integration Test")
    print("=" * 60)
    
    # Test the standalone script
    test_standalone_script()
    
    # Test the database integration
    test_database_integration()
    
    # Show the comparison
    show_comparison()

if __name__ == "__main__":
    main() 