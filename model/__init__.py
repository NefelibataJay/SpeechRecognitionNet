from model.BaseModel import BaseModel
from model.conformer_ctc import ConformerCTC
from model.conformer_attention import ConformerAttention
from model.conformer_transducer import ConformerTransducer

REGISTER_MODEL = {
    "conformer_ctc": ConformerCTC,
    "conformer_attention": ConformerAttention,
    "conformer_transducer": ConformerTransducer
}
