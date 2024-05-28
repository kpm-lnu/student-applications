
class UsersController < ActionController::API
  def index
    users = YAML.load_file("#{Rails.root}/app/controllers/db.yml")
    render json: JSON.dump({name: "Hello, #{params[:name]}", users:})
  end
end
