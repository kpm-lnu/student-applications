require 'google/protobuf'

Google::Protobuf::DescriptorPool.generated_pool.build do
  add_file("protos/helloworld.proto", :syntax => :proto3) do
    add_message "helloworld.HelloRequest" do
      optional :name, :string, 1
    end
    add_message "helloworld.HelloReply" do
      optional :message, :string, 1
      repeated :users, :message, 2, "helloworld.User"
    end
    add_message "helloworld.User" do
      optional :name, :string, 1
      optional :age, :int32, 2
      optional :digest, :string, 3
    end
  end
end

module Helloworld
  HelloRequest = ::Google::Protobuf::DescriptorPool.generated_pool.lookup("helloworld.HelloRequest").msgclass
  HelloReply = ::Google::Protobuf::DescriptorPool.generated_pool.lookup("helloworld.HelloReply").msgclass
  User = ::Google::Protobuf::DescriptorPool.generated_pool.lookup("helloworld.User").msgclass
end
