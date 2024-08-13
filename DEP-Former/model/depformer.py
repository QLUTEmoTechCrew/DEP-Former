from torch import nn
import torch
from model.encoder import Encoder
from model.attention import CrossAttention


class Depformer(nn.Module):
    def __init__(
            self,
            d_k=64,
            d_v=64,
            d_model=224,
            d_ff=32,
            n_heads=8,
            e_layer=3,
            d_layer=2,
            e_stack=3,
            d_feature=256,
            d_mark=224,
            dropout=0.1,
            c=5,
    ):
        super(Depformer, self).__init__()

        self.encoder = Encoder(
            d_k=d_k,
            d_v=d_v,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            n_layer=e_layer,
            n_stack=e_stack,
            d_feature=d_feature,
            d_mark=d_mark,
            dropout=dropout,
            c=c,
            index=0,
        )
        self.encoder_fv = Encoder(
            d_k=d_k,
            d_v=d_v,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            n_layer=e_layer,
            n_stack=e_stack,
            d_feature=128,
            d_mark=d_mark,
            dropout=dropout,
            c=c,
            index=1,
        )
        self.cross = CrossAttention(d_k=d_k,
                                    d_v=d_v,
                                    d_model=d_model,
                                    n_heads=n_heads,
                                    dropout=dropout, )

        self.projection = nn.Linear(d_model, d_feature, bias=True)
        self.fc = nn.Linear(4032, 2)

        self.LN = nn.LayerNorm(1000)
        self.silu = nn.SiLU()
        self.fc1 = nn.Linear(224 * 224, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 128)
        self.fc4 = nn.Linear(4032, 2)

    def Adapter(self, face_data, voice_data):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        face_data = face_data.view(face_data.shape[0], face_data.shape[1], -1).to(device)
        face_data = self.fc1(face_data).to(device)
        face_data = self.LN(face_data).to(device)
        face_data = self.fc3(face_data).to(device)
        face_data = self.silu(face_data).to(device)

        voice_data = voice_data.to(device)
        voice_data = self.fc2(voice_data).to(device)
        voice_data = self.LN(voice_data).to(device)
        voice_data = self.fc3(voice_data).to(device)
        voice_data = self.silu(voice_data).to(device)
        return face_data, voice_data

    def forward(self, face, voice, label):
        face = face.permute(0, 3, 1, 2)
        face_data, voice_data = self.Adapter(face, voice)
        enc_x = torch.cat((face_data, voice_data), dim=2)

        enc_outputs, index, M_out = self.encoder(enc_x)
        enc_outputs_face, index_f, F_out = self.encoder_fv(face_data, index)
        enc_outputs_voice, index_v, V_out = self.encoder_fv(voice_data, index)

        # torch.set_printoptions(threshold=float('inf'))
        # print("Ma_out", M_out)
        # print("Face_out", F_out)
        # print("Voice_out", V_out)
        # torch.set_printoptions(threshold=1000)

        # if index is not None:
        #     for i in range(len(M_out)):
        #         if label[i] == 1:
        #             output_file_path = "/mnt/public/home/wangqx/Yejiayu/depmul2/output_dep_m.txt"
        #             s = M_out[i, :, :10]
        #             with open(output_file_path, 'a') as f:
        #                 f.write('mul')
        #                 torch.set_printoptions(threshold=float('inf'))
        #                 f.write(str(s))
        #                 torch.set_printoptions(threshold=1000)
        #         if label[i] == 0:
        #             output_file_path = "/mnt/public/home/wangqx/Yejiayu/depmul2/output_nor_m.txt"
        #             s = M_out[i, :, :10]
        #             with open(output_file_path, 'a') as f:
        #                 f.write('mul')
        #                 torch.set_printoptions(threshold=float('inf'))
        #                 f.write(str(s))
        #                 torch.set_printoptions(threshold=1000)
        #
        # if index_f is not None:
        #     for i in range(len(F_out)):
        #         if label[i] == 1:
        #             output_file_path = "/mnt/public/home/wangqx/Yejiayu/depmul2/output_dep_f.txt"
        #             s = F_out[i, :, :10]
        #             with open(output_file_path, 'a') as f:
        #                 f.write('f')
        #                 torch.set_printoptions(threshold=float('inf'))
        #                 f.write(str(s))
        #                 torch.set_printoptions(threshold=1000)
        #         if label[i] == 0:
        #             output_file_path = "/mnt/public/home/wangqx/Yejiayu/depmul2/output_nor_f.txt"
        #             s = F_out[i, :, :10]
        #             with open(output_file_path, 'a') as f:
        #                 f.write('f')
        #                 torch.set_printoptions(threshold=float('inf'))
        #                 f.write(str(s))
        #                 torch.set_printoptions(threshold=1000)
        #
        # if index_v is not None:
        #     for i in range(len(V_out)):
        #         if label[i] == 1:
        #             output_file_path = "/mnt/public/home/wangqx/Yejiayu/depmul2/output_dep_v.txt"
        #             s = V_out[i, :, :10]
        #             with open(output_file_path, 'a') as f:
        #                 f.write('v')
        #                 torch.set_printoptions(threshold=float('inf'))
        #                 f.write(str(s))
        #                 torch.set_printoptions(threshold=1000)
        #         if label[i] == 0:
        #             output_file_path = "/mnt/public/home/wangqx/Yejiayu/depmul2/output_nor_v.txt"
        #             s = V_out[i, :, :10]
        #             with open(output_file_path, 'a') as f:
        #                 f.write('v')
        #                 torch.set_printoptions(threshold=float('inf'))
        #                 f.write(str(s))
        #                 torch.set_printoptions(threshold=1000)

        cross_out = self.cross(enc_outputs_face, enc_outputs_voice, enc_outputs, None)

        enc_outputs = self.fc(enc_outputs.view(enc_outputs.shape[0], -1))
        enc_outputs_face = self.fc4(enc_outputs_face.view(enc_outputs_face.shape[0], -1))
        enc_outputs_voice = self.fc4(enc_outputs_voice.view(enc_outputs_voice.shape[0], -1))
        cross_out = self.fc4(cross_out.view(cross_out.shape[0], -1))
        outputs = enc_outputs + 0.1 * cross_out + 0.1 * enc_outputs_face + 0.1 * enc_outputs_voice

        return outputs


