#!/usr/bin/env python3
"""
Test script for minGRU encoder in models/encoders.py
"""

import torch
import sys
import os

# Add project root to path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from models.encoders import RecurrentEncoder

def test_mingru_encoder():
    """Test MinGRU encoder as fallback"""

    batch_size = 4
    seq_length = 16
    feature_dim = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x = torch.randn(batch_size, seq_length, feature_dim, device=device)

    encoder = RecurrentEncoder(
        output_dim=feature_dim,
        num_layers=2,
        num_heads=4,
        rnn_type='mingru',
        is_video_synth_task=True,       # [Charles] video synthesis task experiment
        video_synth_task_out_dim=10,    # [Charles] video synthesis task experiment
        synth_task_rollout_len=1,       # [Charles] ind_head: rollout=1, sel_copy: rollout=10
    ).to(device)
    
    with torch.no_grad():
        initial_states = encoder.get_initial_recurrent_state(batch_size, device)
        logits_rollout = encoder(x, initial_states)

    print(f"MinGRU input shape: {x.shape}")
    print(f"MinGRU output shape: {logits_rollout.shape}")
    print("✓ MinGRU encoder test completed successfully!")
    return True


def test_error_handling():
    """Test error handling for missing xLSTM"""
    print("\n=== Testing Error Handling ===")
    try:
        encoder = RecurrentEncoder(rnn_type='invalid_type')
        print("✗ Should have raised ValueError for invalid rnn_type")
        return False
    except ValueError as e:
        print(f"✓ Correctly caught ValueError: {e}")
    return True


if __name__ == "__main__":
    success = True
    
    try:
        success &= test_mingru_encoder()
        success &= test_error_handling()
        
        if success:
            print("\n🎉 All tests passed!")
        else:
            print("\n❌ Some tests failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)