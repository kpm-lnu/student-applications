this_dir = File.expand_path(File.dirname(__FILE__))
lib_dir = File.join(this_dir, 'lib')
$LOAD_PATH.unshift(lib_dir) unless $LOAD_PATH.include?(lib_dir)

require 'grpc'
require 'helloworld_services_pb'
require 'yaml'

class GreeterServer < Helloworld::Greeter::Service
  def say_hello(hello_req, _unused_call)
    users = YAML.load_file('db.yml')
    Helloworld::HelloReply.new(message: "Hello #{hello_req.name}", users:)
  end
end

def main
  s = GRPC::RpcServer.new
  s.add_http2_port('0.0.0.0:12345', :this_port_is_insecure)
  s.handle(GreeterServer)

  s.run_till_terminated_or_interrupted([1, 'int', 'SIGTERM'])
end

main
