require "google/protobuf"

Google::Protobuf::DescriptorPool.generated_pool.build do
  add_file("onnx.proto", :syntax => :proto2) do
    add_message "onnx.AttributeProto" do
      optional :name, :string, 1
      optional :ref_attr_name, :string, 21
      optional :doc_string, :string, 13
      optional :type, :enum, 20, "onnx.AttributeProto.AttributeType"
      optional :f, :float, 2
      optional :i, :int64, 3
      optional :s, :bytes, 4
      optional :t, :message, 5, "onnx.TensorProto"
      optional :g, :message, 6, "onnx.GraphProto"
      optional :sparse_tensor, :message, 22, "onnx.SparseTensorProto"
      repeated :floats, :float, 7
      repeated :ints, :int64, 8
      repeated :strings, :bytes, 9
      repeated :tensors, :message, 10, "onnx.TensorProto"
      repeated :graphs, :message, 11, "onnx.GraphProto"
      repeated :sparse_tensors, :message, 23, "onnx.SparseTensorProto"
    end
    add_enum "onnx.AttributeProto.AttributeType" do
      value :UNDEFINED, 0
      value :FLOAT, 1
      value :INT, 2
      value :STRING, 3
      value :TENSOR, 4
      value :GRAPH, 5
      value :SPARSE_TENSOR, 11
      value :FLOATS, 6
      value :INTS, 7
      value :STRINGS, 8
      value :TENSORS, 9
      value :GRAPHS, 10
      value :SPARSE_TENSORS, 12
    end
    add_message "onnx.ValueInfoProto" do
      optional :name, :string, 1
      optional :type, :message, 2, "onnx.TypeProto"
      optional :doc_string, :string, 3
    end
    add_message "onnx.NodeProto" do
      repeated :input, :string, 1
      repeated :output, :string, 2
      optional :name, :string, 3
      optional :op_type, :string, 4
      optional :domain, :string, 7
      repeated :attribute, :message, 5, "onnx.AttributeProto"
      optional :doc_string, :string, 6
    end
    add_message "onnx.ModelProto" do
      optional :ir_version, :int64, 1
      repeated :opset_import, :message, 8, "onnx.OperatorSetIdProto"
      optional :producer_name, :string, 2
      optional :producer_version, :string, 3
      optional :domain, :string, 4
      optional :model_version, :int64, 5
      optional :doc_string, :string, 6
      optional :graph, :message, 7, "onnx.GraphProto"
      repeated :metadata_props, :message, 14, "onnx.StringStringEntryProto"
    end
    add_message "onnx.StringStringEntryProto" do
      optional :key, :string, 1
      optional :value, :string, 2
    end
    add_message "onnx.TensorAnnotation" do
      optional :tensor_name, :string, 1
      repeated :quant_parameter_tensor_names, :message, 2, "onnx.StringStringEntryProto"
    end
    add_message "onnx.GraphProto" do
      repeated :node, :message, 1, "onnx.NodeProto"
      optional :name, :string, 2
      repeated :initializer, :message, 5, "onnx.TensorProto"
      repeated :sparse_initializer, :message, 15, "onnx.SparseTensorProto"
      optional :doc_string, :string, 10
      repeated :input, :message, 11, "onnx.ValueInfoProto"
      repeated :output, :message, 12, "onnx.ValueInfoProto"
      repeated :value_info, :message, 13, "onnx.ValueInfoProto"
      repeated :quantization_annotation, :message, 14, "onnx.TensorAnnotation"
    end
    add_message "onnx.TensorProto" do
      repeated :dims, :int64, 1
      optional :data_type, :int32, 2
      optional :segment, :message, 3, "onnx.TensorProto.Segment"
      repeated :float_data, :float, 4
      repeated :int32_data, :int32, 5
      repeated :string_data, :bytes, 6
      repeated :int64_data, :int64, 7
      optional :name, :string, 8
      optional :doc_string, :string, 12
      optional :raw_data, :bytes, 9
      repeated :external_data, :message, 13, "onnx.StringStringEntryProto"
      optional :data_location, :enum, 14, "onnx.TensorProto.DataLocation"
      repeated :double_data, :double, 10
      repeated :uint64_data, :uint64, 11
    end
    add_message "onnx.TensorProto.Segment" do
      optional :begin, :int64, 1
      optional :end, :int64, 2
    end
    add_enum "onnx.TensorProto.DataType" do
      value :UNDEFINED, 0
      value :FLOAT, 1
      value :UINT8, 2
      value :INT8, 3
      value :UINT16, 4
      value :INT16, 5
      value :INT32, 6
      value :INT64, 7
      value :STRING, 8
      value :BOOL, 9
      value :FLOAT16, 10
      value :DOUBLE, 11
      value :UINT32, 12
      value :UINT64, 13
      value :COMPLEX64, 14
      value :COMPLEX128, 15
      value :BFLOAT16, 16
    end
    add_enum "onnx.TensorProto.DataLocation" do
      value :DEFAULT, 0
      value :EXTERNAL, 1
    end
    add_message "onnx.SparseTensorProto" do
      optional :values, :message, 1, "onnx.TensorProto"
      optional :indices, :message, 2, "onnx.TensorProto"
      repeated :dims, :int64, 3
    end
    add_message "onnx.TensorShapeProto" do
      repeated :dim, :message, 1, "onnx.TensorShapeProto.Dimension"
    end
    add_message "onnx.TensorShapeProto.Dimension" do
      optional :denotation, :string, 3
      oneof :value do
        optional :dim_value, :int64, 1
        optional :dim_param, :string, 2
      end
    end
    add_message "onnx.TypeProto" do
      optional :denotation, :string, 6
      oneof :value do
        optional :tensor_type, :message, 1, "onnx.TypeProto.Tensor"
      end
    end
    add_message "onnx.TypeProto.Tensor" do
      optional :elem_type, :int32, 1
      optional :shape, :message, 2, "onnx.TensorShapeProto"
    end
    add_message "onnx.TypeProto.SparseTensor" do
      optional :elem_type, :int32, 1
      optional :shape, :message, 2, "onnx.TensorShapeProto"
    end
    add_message "onnx.OperatorSetIdProto" do
      optional :domain, :string, 1
      optional :version, :int64, 2
    end
    add_enum "onnx.Version" do
      value :START_VERSION, 0
      value :IR_VERSION_2017_10_10, 1
      value :IR_VERSION_2017_10_30, 2
      value :IR_VERSION_2017_11_3, 3
      value :IR_VERSION_2019_1_22, 4
      value :IR_VERSION_2019_3_18, 5
      value :IR_VERSION, 6
    end
  end
end

module Eps
  module Onnx
    AttributeProto = Google::Protobuf::DescriptorPool.generated_pool.lookup("onnx.AttributeProto").msgclass
    AttributeProto::AttributeType = Google::Protobuf::DescriptorPool.generated_pool.lookup("onnx.AttributeProto.AttributeType").enummodule
    ValueInfoProto = Google::Protobuf::DescriptorPool.generated_pool.lookup("onnx.ValueInfoProto").msgclass
    NodeProto = Google::Protobuf::DescriptorPool.generated_pool.lookup("onnx.NodeProto").msgclass
    ModelProto = Google::Protobuf::DescriptorPool.generated_pool.lookup("onnx.ModelProto").msgclass
    StringStringEntryProto = Google::Protobuf::DescriptorPool.generated_pool.lookup("onnx.StringStringEntryProto").msgclass
    TensorAnnotation = Google::Protobuf::DescriptorPool.generated_pool.lookup("onnx.TensorAnnotation").msgclass
    GraphProto = Google::Protobuf::DescriptorPool.generated_pool.lookup("onnx.GraphProto").msgclass
    TensorProto = Google::Protobuf::DescriptorPool.generated_pool.lookup("onnx.TensorProto").msgclass
    TensorProto::Segment = Google::Protobuf::DescriptorPool.generated_pool.lookup("onnx.TensorProto.Segment").msgclass
    TensorProto::DataType = Google::Protobuf::DescriptorPool.generated_pool.lookup("onnx.TensorProto.DataType").enummodule
    TensorProto::DataLocation = Google::Protobuf::DescriptorPool.generated_pool.lookup("onnx.TensorProto.DataLocation").enummodule
    SparseTensorProto = Google::Protobuf::DescriptorPool.generated_pool.lookup("onnx.SparseTensorProto").msgclass
    TensorShapeProto = Google::Protobuf::DescriptorPool.generated_pool.lookup("onnx.TensorShapeProto").msgclass
    TensorShapeProto::Dimension = Google::Protobuf::DescriptorPool.generated_pool.lookup("onnx.TensorShapeProto.Dimension").msgclass
    TypeProto = Google::Protobuf::DescriptorPool.generated_pool.lookup("onnx.TypeProto").msgclass
    TypeProto::Tensor = Google::Protobuf::DescriptorPool.generated_pool.lookup("onnx.TypeProto.Tensor").msgclass
    TypeProto::SparseTensor = Google::Protobuf::DescriptorPool.generated_pool.lookup("onnx.TypeProto.SparseTensor").msgclass
    OperatorSetIdProto = Google::Protobuf::DescriptorPool.generated_pool.lookup("onnx.OperatorSetIdProto").msgclass
    Version = Google::Protobuf::DescriptorPool.generated_pool.lookup("onnx.Version").enummodule
  end
end
