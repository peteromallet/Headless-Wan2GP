# VID2VID INITIALIZATION PATCH
# Apply to models/wan/any2video.py

# 1. Add these parameters to the generate() function signature (after video_prompt_type):
"""
        latent_noise_mask_strength = 0.0,  # 0.0 = disabled, 1.0 = full latent noise masking
        vid2vid_init_video = None,  # Path to video for vid2vid initialization (gap frames)
        vid2vid_init_strength = 0.7,  # 0.0 = pure vid2vid (keep original), 1.0 = pure txt2vid (random noise)
"""

# 2. Add this block AFTER latents are initialized (after: latents = torch.randn(...))
# but BEFORE the apg_switch block:
VID2VID_INIT_CODE = '''
        # Vid2vid initialization: Use provided video as starting point instead of pure noise
        # This is useful for VACE replace mode where we want to refine existing frames
        if vid2vid_init_video is not None and vid2vid_init_strength < 1.0:
            try:
                import cv2
                # NOTE:
                # numpy is already imported at module scope as `np`.
                # Re-importing it here makes `np` a *local* variable in this whole function,
                # which can crash earlier code paths (e.g. euler timesteps) with:
                #   UnboundLocalError: local variable 'np' referenced before assignment
                
                print(f"[VID2VID_INIT] Loading video for initialization: {vid2vid_init_video}")
                print(f"[VID2VID_INIT] Strength: {vid2vid_init_strength} (0=keep original, 1=random noise)")
                
                # Load video frames
                cap = cv2.VideoCapture(str(vid2vid_init_video))
                vid2vid_frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Convert BGR to RGB and normalize to [-1, 1]
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_tensor = torch.from_numpy(frame_rgb).float().div_(127.5).sub_(1)
                    vid2vid_frames.append(frame_tensor)
                cap.release()
                
                if len(vid2vid_frames) > 0:
                    # Stack frames: [F, H, W, C] -> [C, F, H, W]
                    vid2vid_tensor = torch.stack(vid2vid_frames, dim=0).permute(3, 0, 1, 2).to(self.device)
                    print(f"[VID2VID_INIT] Loaded {len(vid2vid_frames)} frames, shape: {vid2vid_tensor.shape}")
                    
                    # Encode with VAE
                    with torch.no_grad():
                        vid2vid_latents = self.vae.encode([vid2vid_tensor], tile_size=VAE_tile_size)[0]
                    print(f"[VID2VID_INIT] Encoded to latent shape: {vid2vid_latents.shape}")
                    
                    # Handle frame count mismatch between vid2vid video and target
                    target_lat_frames = target_shape[1]  # Latent frame count
                    vid2vid_lat_frames = vid2vid_latents.shape[1]
                    
                    if vid2vid_lat_frames != target_lat_frames:
                        print(f"[VID2VID_INIT] Frame count mismatch: vid2vid has {vid2vid_lat_frames} latent frames, target has {target_lat_frames}")
                        if vid2vid_lat_frames > target_lat_frames:
                            # Trim excess frames
                            vid2vid_latents = vid2vid_latents[:, :target_lat_frames]
                        else:
                            # Pad with random noise for remaining frames
                            pad_size = target_lat_frames - vid2vid_lat_frames
                            padding = torch.randn(vid2vid_latents.shape[0], pad_size, *vid2vid_latents.shape[2:], 
                                                  device=self.device, dtype=vid2vid_latents.dtype)
                            vid2vid_latents = torch.cat([vid2vid_latents, padding], dim=1)
                    
                    # Add batch dimension if needed
                    if vid2vid_latents.dim() == 4:
                        vid2vid_latents = vid2vid_latents.unsqueeze(0)
                    
                    # Blend: latents = strength * noise + (1 - strength) * encoded
                    # Higher strength = more noise = more regeneration
                    # Lower strength = more original = less change
                    print(f"[VID2VID_INIT] Blending latents with strength {vid2vid_init_strength}")
                    latents = vid2vid_init_strength * latents + (1.0 - vid2vid_init_strength) * vid2vid_latents.to(latents.dtype)
                    
                    print(f"[VID2VID_INIT] Vid2vid initialization complete, final latent shape: {latents.shape}")
                else:
                    print(f"[VID2VID_INIT] Warning: Could not load any frames from {vid2vid_init_video}")
                    
            except Exception as e:
                print(f"[VID2VID_INIT] Error during vid2vid initialization: {e}")
                import traceback
                traceback.print_exc()
                # Continue with random latents on error
'''

# 3. Add LATENT NOISE MASK initialization block after VACE encoding (after z = self.vace_latent(z0, m0)):
LATENT_NOISE_MASK_CODE = '''
            # Latent noise mask: Store original latents and mask for blending during denoising
            # z0[0] shape: [32, frames, h, w] where first 16 channels are inactive, last 16 are reactive
            # m0[0] shape: [64, frames, h, w] - the mask in latent space (0=preserve, 1=generate)
            latent_noise_mask_original = None
            latent_noise_mask_blend = None
            latent_noise_mask_noise = None
            if latent_noise_mask_strength > 0:
                # Get the inactive latents (first 16 channels of z0) - these are the preserved regions
                latent_noise_mask_original = z0[0][:16].clone().unsqueeze(0)  # [1, 16, frames, h, w]
                # Get the mask from m0 - average across the 64 channels to get a single mask
                # m0 values: 0 = preserve (black in mask video), 1 = generate (white in mask video)
                latent_noise_mask_blend = m0[0].mean(dim=0, keepdim=True).unsqueeze(0)  # [1, 1, frames, h, w]
                # Store noise once for consistent blending across all denoising steps
                latent_noise_mask_noise = torch.randn_like(latent_noise_mask_original)
                print(f"[LATENT_NOISE_MASK] Enabled with strength={latent_noise_mask_strength}")
                print(f"[LATENT_NOISE_MASK] Original latents shape: {latent_noise_mask_original.shape}")
                print(f"[LATENT_NOISE_MASK] Mask blend shape: {latent_noise_mask_blend.shape}")
'''

