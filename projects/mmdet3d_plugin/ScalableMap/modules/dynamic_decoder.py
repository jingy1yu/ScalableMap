import torch
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmdet.models.utils.transformer import inverse_sigmoid


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DynamicDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default： `LN`.
    """
    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(DynamicDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False
        self.model_dimension = 256
        self.instances_num = 50

    def forward(self,
                query,
                query_pos,
                *args,
                reference_points=None,
                reg_branches=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        insert_count = 0
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points[..., :2].unsqueeze(2)
            output = layer(
                output,
                *args,
                query_pos=query_pos,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            intermediate.append(output)
            intermediate_reference_points.append(reference_points)

            if lid in [0, 1, 2]:
                insert_count += 1
                num_query, bs, _ = output.shape
                num_control = num_query // self.instances_num
                reference_points = reference_points.view(-1, self.instances_num, num_control, 2)
                insert_reference_points = (torch.add(reference_points,
                                                     torch.roll(reference_points, -1, dims=2)))[..., :-1, :] * 0.5
                insert_reference_points = torch.stack((reference_points[..., :-1, :],
                                                       insert_reference_points), dim=3).view(bs, self.instances_num, -1,
                                                                                             2)
                reference_points = torch.cat((insert_reference_points,
                                              reference_points[..., -1, :].unsqueeze(-2)), dim=-2).reshape(bs, -1, 2)
                query_pos = query_pos.view(self.instances_num, num_control, bs, self.model_dimension)
                insert_query_pos = (torch.add(query_pos, torch.roll(query_pos, -1, dims=1)))[:, :-1, ...] * 0.5

                insert_query_pos = torch.stack((query_pos[:, :-1, ...],
                                         insert_query_pos), dim=2).view(self.instances_num, -1, bs, self.model_dimension)
                query_pos = torch.cat((insert_query_pos,
                                    query_pos[:, -1, ...].unsqueeze(1)), dim=1).reshape(-1, bs, self.model_dimension)
                output = output.view(self.instances_num, num_control, bs, self.model_dimension)
                insert_tgt = (torch.add(output, torch.roll(output, -1, dims=1)))[:, :-1, ...] * 0.5
                insert_tgt = torch.stack((output[:, :-1, ...],
                                          insert_tgt), dim=2).view(self.instances_num, -1, bs, self.model_dimension)
                output = torch.cat((insert_tgt,
                                    output[:, -1, ...].unsqueeze(1)), dim=1).reshape(-1, bs, self.model_dimension)

        return intermediate, intermediate_reference_points

