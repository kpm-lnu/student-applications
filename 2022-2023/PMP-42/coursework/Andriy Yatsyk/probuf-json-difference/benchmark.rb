require 'benchmark/ips'
require 'yajl'
require 'oj'
require 'json'
require_relative 'user_pb'

data = {
  username: 'johndoe',
  email: 'johndoe@example.com',
  date_joined: '2023-05-02T12:30:00Z',
  is_active: true,
  profile: {
    full_name: 'John Doe',
    age: 20,
    address: '123 Main St, Anytown, USA',
    phone_number: '+1-555-123-4567'
  },
  posts: [
    {
      post_id: 1,
      title: 'My first blog post',
      content: 'This is the content of my first blog post.',
      date_created: '2023-04-01T14:00:00Z',
      likes: 10,
      tags: ['blog', 'first_post', 'welcome'],
      comments: [
        {
          comment_id: 101,
          author: 'Jane',
          content: 'Great first post!',
          date_created: '2023-04-01T15:00:00Z',
          likes: 3
        },
        {
          comment_id: 102,
          author: 'Mark',
          content: 'Welcome to the blogging world!',
          date_created: '2023-04-01T16:00:00Z',
          likes: 4
        },
        {
          comment_id: 103,
          author: 'Sara',
          content: 'Looking forward to more posts!',
          date_created: '2023-04-01T17:00:00Z',
          likes: 2
        },
        {
          comment_id: 104,
          author: 'Tom',
          content: 'Nice start!',
          date_created: '2023-04-01T18:00:00Z',
          likes: 1
        }
      ]
    },
    {
      post_id: 2,
      title: 'A day in the life',
      content: 'Today, I will share my daily routine.',
      date_created: '2023-04-02T10:00:00Z',
      likes: 5,
      tags: ['daily', 'routine', 'life'],
      comments: [
        {
          comment_id: 201,
          author: 'Lucy',
          content: 'Interesting routine!',
          date_created: '2023-04-02T11:00:00Z',
          likes: 2
        },
        {
          comment_id: 202,
          author: 'Mike',
          content: 'Thanks for sharing!',
          date_created: '2023-04-02T12:00:00Z',
          likes: 3
        },
        {
          comment_id: 203,
          author: 'Emma',
          content: 'I have a similar routine.',
          date_created: '2023-04-02T13:00:00Z',
          likes: 1
        },
        {
          comment_id: 204,
          author: 'Sam',
          content: 'What do you do on weekends?',
          date_created: '2023-04-02T14:00:00Z',
          likes: 2
        },
        {
          comment_id: 205,
          author: 'Anna',
          content: 'Your routine is inspiring!',
          date_created: '2023-04-02T15:00:00Z',
          likes: 3
        }
      ]
    }
  ]
}

proto_model = User.new(data)
proto_encoded_data = User.encode(proto_model)
json_encoded_data = JSON.dump(data)

puts "JSON payload bytesize #{json_encoded_data.bytesize}"
puts "Protobuf payload bytesize #{proto_encoded_data.bytesize}"
puts
puts 'Encoding...'
puts
Benchmark.ips do |x|
  x.config(time: 10)

  x.report('Yajl encoding') do
    Yajl::Encoder.encode(data)
  end

  x.report('Oj encoding') do
    Oj.dump(data)
  end

  x.report('standard JSON encoding') do
    JSON.dump(data)
  end

  x.report('protobuf encoding') do
    User.encode(proto_model)
  end

  x.report('protobuf with model init') do
    User.new(data).to_proto
  end

  x.compare!
end
puts
puts 'Decoding...'
puts
Benchmark.ips do |x|
  x.config(time: 10)

  x.report('Yajl parsing') do
    Yajl::Parser.parse(json_encoded_data)
  end

  x.report('Oj parsing') do
    Oj.load(json_encoded_data)
  end

  x.report('standard JSON parsing') do
    JSON.parse(json_encoded_data)
  end

  x.report('protobuf parsing') do
    User.decode(proto_encoded_data)
  end

  x.compare!
end

