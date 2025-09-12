from abc import ABC, abstractmethod
import torch.nn as nn
import torch

from models.modules.unets.cnn_decoder import MultiFrameUNet
from models.modules.unets.audio_decoder import MultiAudioUNet
from models.modules.unets.action_decoder import MultiKeyboardUNet, MultiMouseUNet
from models.encoders import StateEncoder, ActionEncoder, StateActionEncoder
from models.utils import MODALITY_TO_LATENT


def _latent_keys(names):
    # names must be utils’ modality keys, e.g. "video", "key_press", etc.
    return [MODALITY_TO_LATENT[n] for n in names]

def _make_stm_dict(z, latent_keys):
    if not latent_keys:
        return None
    d = {}
    for k in latent_keys:
        x = z[k]
        d[k] = x.flatten(2) if x.dim() > 3 else x   # [B,T,...] → [B,T,D]
    return d

class MultiModalModel(nn.Module, ABC):
    def __init__(self, args, video_latent_shape=(8, 96, 160), audio_in_shape=(15, 128),
                 audio_out_shape=(15, 128), keyboard_shape=(2*5, 16), mouse_shape=(2*10, 2)):
        super().__init__()
        self.args = args
        self.h_dim = self.args.h_dim
        self.out_layers = args.out_layers
        self.video_latent_shape = video_latent_shape
        self.audio_in_shape = audio_in_shape
        self.audio_out_shape = audio_out_shape
        self.keyboard_shape = keyboard_shape
        self.mouse_shape = mouse_shape

        self._init_encoders()
        self._init_heads()

    @abstractmethod
    def _init_encoders(self):
        pass

    @abstractmethod
    def _init_heads(self):
        pass

    @staticmethod
    def add_command_line_options(argparser):
        argparser.add_argument('--h_dim', default=128, type=int)
        argparser.add_argument('--out_layers', default=4, type=int)
        
        p = argparser.add_argument
        p('--frame_head_stm',     nargs='*', default=['key_press', 'mouse_movement', 'audio_out'])
        p('--audio_out_head_stm', nargs='*', default=['key_press', 'mouse_movement', 'video'])
        p('--audio_in_head_stm',  nargs='*', default=['audio_out'])
        p('--keyboard_head_stm',  nargs='*', default=['video'])
        p('--mouse_head_stm',     nargs='*', default=['video'])

    @abstractmethod
    def forward(self, z, noise_encoding, obs_encoding):
        '''
            z: dict of noisy diffusion variables of shape [B, ...]
            noise_encoding: noise embedding of shape [B, D]
            obs_encoding: observation embedding of shape [B, D]
        '''
        pass

    def get_states_actions_encoding(self, input_dict):
        '''
            input_dict containing:
                frame_latent: [B, c, h, w]
                audio_in_latent: [B, l, d]
                audio_out_latent: [B, l, d]
                keyboard_latent: [B, d]
                mouse_latent: [B, d]
            scale_factor: [B, 1]
        '''
        frame_latent = input_dict['frame_latent'].flatten(0, 1)
        audio_out_latent = input_dict['audio_out_latent'].flatten(0, 1)
        audio_in_latent = input_dict['audio_in_latent'].flatten(0, 1)
        keyboard_latent = input_dict['keyboard_latent'].flatten(0, 1)
        mouse_latent = input_dict['mouse_latent'].flatten(0, 1)

        gc = bool(getattr(self.args, "gradient_checkpointing", 0))

        state_enc = self.state_encoder(frame_latent, audio_out_latent, gradient_checkpoint=gc)
        action_enc = self.action_encoder(audio_in_latent, keyboard_latent, mouse_latent, gradient_checkpoint=gc)
        timestep_enc = self.state_action_encoder(state_enc, action_enc, gradient_checkpoint=gc)
        return timestep_enc


class MultiModalTemporalUNet(MultiModalModel):
    """
    All-modality UNet that:
      • keeps the per-modality temporal UNets (MultiFrame, MultiAudio, etc.)
      • still encodes (state, action) via the original encoders
      • allows every head to cross-attend to the others through modality_stm_dict
    """

    # ----------------------------- BUILDERS --------------------------- #
    def _init_encoders(self):
        self.state_encoder = StateEncoder(output_dim=self.h_dim, frame_encoder='transformer')
        self.action_encoder = ActionEncoder(output_dim=self.h_dim)
        self.state_action_encoder = StateActionEncoder(output_dim=self.h_dim)

    def _init_heads(self):
        # ---- temporal UNets with obs-mask input --------------------- #
        self.frame_denoiser_head = MultiFrameUNet(
            self.args, input_dim=self.video_latent_shape[0], h_dim=self.h_dim,
        )
        self.audio_in_denoiser_head = MultiAudioUNet(
            self.args, input_dim=self.audio_in_shape[1], h_dim=self.h_dim,
        )
        self.audio_out_denoiser_head = MultiAudioUNet(
            self.args, input_dim=self.audio_out_shape[1], h_dim=self.h_dim,
        )
        self.keyboard_denoiser_head = MultiKeyboardUNet(
            self.args, input_dim=self.keyboard_shape[1], h_dim=self.h_dim,
        )
        self.mouse_denoiser_head = MultiMouseUNet(
            self.args, input_dim=self.mouse_shape[1], h_dim=self.h_dim,
        )

    # ------------------------- CLI helper ---------------------------- #
    @staticmethod
    def add_command_line_options(p):
        MultiModalModel.add_command_line_options(p)
        MultiFrameUNet.add_command_line_options(p)
        MultiAudioUNet.add_command_line_options(p)
        MultiKeyboardUNet.add_command_line_options(p)
        MultiMouseUNet.add_command_line_options(p)

    # --------------------------- FORWARD ----------------------------- #
    def forward(self, z, emb,
                valid_mask, obs_mask, ctx_vec=None):
        """
        z : dict with keys
              frame_latent    → [B,T,C,H,W]
              audio_in_latent → [B,T,L,D]  (transposed inside)
              audio_out_latent→ [B,T,L,D]
              keyboard_latent → [B,T,10,16]
              mouse_latent    → [B,T,20, 2]
        valid_mask, obs_mask : [B,T]
        """
        if self.args.enable_modality_stm:
            frame_stm_dict = _make_stm_dict(z, _latent_keys(self.args.frame_head_stm))
            audio_in_stm_dict = _make_stm_dict(z, _latent_keys(self.args.audio_in_head_stm))
            audio_out_stm_dict = _make_stm_dict(z, _latent_keys(self.args.audio_out_head_stm))
            keyboard_stm_dict = _make_stm_dict(z, _latent_keys(self.args.keyboard_head_stm))
            mouse_stm_dict = _make_stm_dict(z, _latent_keys(self.args.mouse_head_stm))
        else:
            frame_stm_dict = audio_in_stm_dict = audio_out_stm_dict = keyboard_stm_dict = mouse_stm_dict = None

        out = dict()

        # ---------------- VIDEO -------------------------------------- #
        out['frame_latent'] = self.frame_denoiser_head(
            z['frame_latent'], emb, valid_mask, obs_mask, modality_stm_dict = frame_stm_dict, ctx_vec=ctx_vec
        )

        # ---------------- AUDIO (in / out) --------------------------- #
        # permute to [B,T,C,L]
        audio_in = z['audio_in_latent'].transpose(-1, -2)
        audio_out= z['audio_out_latent'].transpose(-1, -2)

        out['audio_in_latent'] = self.audio_in_denoiser_head(
            audio_in, emb, valid_mask, obs_mask, modality_stm_dict = audio_in_stm_dict, ctx_vec=ctx_vec
        ).transpose(-1, -2)

        out['audio_out_latent']= self.audio_out_denoiser_head(
            audio_out, emb, valid_mask, obs_mask, modality_stm_dict = audio_out_stm_dict, ctx_vec=ctx_vec
        ).transpose(-1, -2)

        # ---------------- KEYBOARD / MOUSE --------------------------- #
        out['keyboard_latent'] = self.keyboard_denoiser_head(
            z['keyboard_latent'], emb, valid_mask, obs_mask, modality_stm_dict = keyboard_stm_dict, ctx_vec=ctx_vec
        )

        out['mouse_latent']    = self.mouse_denoiser_head(
            z['mouse_latent'], emb, valid_mask, obs_mask, modality_stm_dict = mouse_stm_dict, ctx_vec=ctx_vec
        )

        return out


class ActionConditionedWorldModelUnet(MultiModalModel):
    """
    All-modality UNet that:
      • keeps the per-modality temporal UNets (MultiFrame, MultiAudio, etc.)
      • still encodes (state, action) via the original encoders
      • allows every head to cross-attend to the others through modality_stm_dict
    """

    # ----------------------------- BUILDERS --------------------------- #
    def _init_encoders(self):
        self.state_encoder = StateEncoder(output_dim=self.h_dim, frame_encoder='transformer')
        self.action_encoder = ActionEncoder(output_dim=self.h_dim)
        self.state_action_encoder = StateActionEncoder(output_dim=self.h_dim)

    def _init_heads(self):
        # ---- temporal UNets with obs-mask input --------------------- #
        self.frame_denoiser_head = MultiFrameUNet(
            self.args, input_dim=self.video_latent_shape[0], h_dim=self.h_dim,
        ).to(memory_format=torch.channels_last)
        self.audio_out_denoiser_head = MultiAudioUNet(
            self.args, input_dim=self.audio_out_shape[1], h_dim=self.h_dim,
        )

    # ------------------------- CLI helper ---------------------------- #
    @staticmethod
    def add_command_line_options(p):
        MultiModalModel.add_command_line_options(p)
        MultiFrameUNet.add_command_line_options(p)
        MultiAudioUNet.add_command_line_options(p)

    # --------------------------- FORWARD ----------------------------- #
    def forward(self, z, emb, valid_mask, obs_mask, ctx_vec=None):
        """
        z : dict with keys
              frame_latent    → [B,T,C,H,W]
              audio_in_latent → [B,T,L,D]  (transposed inside)
              audio_out_latent→ [B,T,L,D]
              keyboard_latent → [B,T,10,16]
              mouse_latent    → [B,T,20, 2]
        valid_mask, obs_mask : [B,T]
        """
        if self.args.enable_modality_stm:
            frame_stm_dict = _make_stm_dict(z, _latent_keys(self.args.frame_head_stm))
            audio_out_stm_dict = _make_stm_dict(z, _latent_keys(self.args.audio_out_head_stm))
        else:
            frame_stm_dict = audio_out_stm_dict = None

        out = dict()

        # ---------------- VIDEO -------------------------------------- #
        out['frame_latent'] = self.frame_denoiser_head(
            z['frame_latent'], emb, valid_mask, obs_mask, modality_stm_dict = frame_stm_dict, ctx_vec=ctx_vec
        )

        # ---------------- AUDIO (out) --------------------------- #
        # permute to [B,T,D,L]
        audio_out= z['audio_out_latent'].transpose(-1, -2)

        out['audio_out_latent']= self.audio_out_denoiser_head(
            audio_out, emb, valid_mask, obs_mask, modality_stm_dict = audio_out_stm_dict, ctx_vec=ctx_vec
        ).transpose(-1, -2)

        return out
    
    
class ActionModelUnet(MultiModalModel):
    def _init_encoders(self):
        self.state_encoder = StateEncoder(output_dim=self.h_dim, frame_encoder='transformer')
        self.action_encoder = ActionEncoder(output_dim=self.h_dim)
        self.state_action_encoder = StateActionEncoder(output_dim=self.h_dim)

    def _init_heads(self):
        self.audio_in_denoiser_head = MultiAudioUNet(
            self.args, input_dim=self.audio_in_shape[1], h_dim=self.h_dim,
        )
        self.keyboard_denoiser_head = MultiKeyboardUNet(
            self.args, input_dim=self.keyboard_shape[1], h_dim=self.h_dim,
        )
        self.mouse_denoiser_head = MultiMouseUNet(
            self.args, input_dim=self.mouse_shape[1], h_dim=self.h_dim,
        )

    # ------------------------- CLI helper ---------------------------- #
    @staticmethod
    def add_command_line_options(p):
        MultiModalModel.add_command_line_options(p)
        MultiAudioUNet.add_command_line_options(p)
        MultiKeyboardUNet.add_command_line_options(p)
        MultiMouseUNet.add_command_line_options(p)

    # --------------------------- FORWARD ----------------------------- #
    def forward(self, z, emb, valid_mask, obs_mask, ctx_vec=None):
        """
        z : dict with keys
              frame_latent    → [B,T,C,H,W]
              audio_in_latent → [B,T,L,D]  (transposed inside)
              audio_out_latent→ [B,T,L,D]
              keyboard_latent → [B,T,10,16]
              mouse_latent    → [B,T,20, 2]
        valid_mask, obs_mask : [B,T]
        """
        if self.args.enable_modality_stm:
            audio_in_stm_dict = _make_stm_dict(z, _latent_keys(self.args.audio_in_head_stm))
            keyboard_stm_dict = _make_stm_dict(z, _latent_keys(self.args.keyboard_head_stm))
            mouse_stm_dict = _make_stm_dict(z, _latent_keys(self.args.mouse_head_stm))
        else:
            audio_in_stm_dict = keyboard_stm_dict = mouse_stm_dict = None

        out = dict()

        out['keyboard_latent'] = self.keyboard_denoiser_head(
            z['keyboard_latent'], emb, valid_mask, obs_mask, modality_stm_dict = keyboard_stm_dict, ctx_vec=ctx_vec
        )

        out['mouse_latent']    = self.mouse_denoiser_head(
            z['mouse_latent'], emb, valid_mask, obs_mask, modality_stm_dict = mouse_stm_dict, ctx_vec=ctx_vec
        )

        audio_in= z['audio_in_latent'].transpose(-1, -2)
        out['audio_in_latent']= self.audio_in_denoiser_head(
            audio_in, emb, valid_mask, obs_mask, modality_stm_dict = audio_in_stm_dict, ctx_vec=ctx_vec
        ).transpose(-1, -2)

        return out