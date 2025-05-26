#ifndef FEM_ENGINE_ELEMENT_H
#define FEM_ENGINE_ELEMENT_H

#include "types.h"
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

const unsigned int N = 4; // 4 bytes for flat

/**
 * @class Element
 * @brief Represents a graphical element in the engine, supporting various types such as scalars, vectors, arrays, and structures.
 *
 * The Element class supports hierarchical composition of sub-elements using a Composite pattern.
 * Builders are provided for constructing specific types of elements in a structured manner.
 */
class Element {
public:
    Type type; ///< The type of the element (e.g., SCALAR, VEC2, etc.)
    unsigned int base_align; ///< Base alignment requirement for the element.
    unsigned int length; ///< Length of the array or number of elements in a structure.
    std::vector<std::shared_ptr<Element>> list; ///< Sub-elements for structs or arrays.

    /**
     * @brief Default constructor initializing a scalar element.
     * @param type The type of the element.
     */
    Element(Type type = Type::SCALAR);

    /**
     * @brief Converts the element type to a string representation.
     * @return A string representing the type.
     */
    std::string type_str() const;

    /**
     * @brief Calculates the power-of-2 alignment of the element.
     * @return The power-of-2 alignment.
     */
    unsigned int align_pow2() const;

    /**
     * @brief Calculates the size of the element without padding.
     * @return The size in bytes.
     */
    unsigned int calc_size() const;

    /**
     * @brief Calculates the padded size of the element.
     * @return The size in bytes, including padding.
     */
    unsigned int calc_padded_size() const;

    // Builder classes
    class Builder;
    class ScalarBuilder;
    class VecBuilder;
    class ArrayBuilder;
    class ColMatBuilder;
    class RowMatBuilder;
    class StructBuilder;

    /**
     * @class Factory
     * @brief Factory class for creating instances of Element with predefined types.
     *
     */
    class Factory {
    private:
        Factory() = default; ///< Private constructor for Singleton pattern.
        static std::shared_ptr<Factory> instance; ///< Singleton instance.
    public:
        static std::shared_ptr<Factory> get_instance(); ///< Retrieves the Singleton instance.

        std::shared_ptr<Element> create_scalar();
        std::shared_ptr<Element> create_vector(unsigned char dim);
        std::shared_ptr<Element> create_array(unsigned int len, const std::shared_ptr<Element>& elem);
        std::shared_ptr<Element> create_struct(const std::vector<std::shared_ptr<Element>>& subelements);
    };
};

// Abstract Builder base class
class Element::Builder {
public:
    virtual ~Builder() = default;
    virtual std::shared_ptr<Element> build() const = 0;
};

class Element::ScalarBuilder : public Builder {
public:
    std::shared_ptr<Element> build() const override {
        return std::make_shared<Element>(Type::SCALAR);
    }
};

class Element::VecBuilder : public Builder {
private:
    unsigned char dim;
public:
    explicit VecBuilder(unsigned char dimension) : dim(dimension) {}

    std::shared_ptr<Element> build() const override {
        switch (dim) {
            case 2: return std::make_shared<Element>(Type::VEC2);
            case 3: return std::make_shared<Element>(Type::VEC3);
            case 4:
            default:
                return std::make_shared<Element>(Type::VEC4);
        }
    }
};

class Element::ArrayBuilder : public Builder {
private:
    unsigned int length;
    std::shared_ptr<Element> element;
public:
    ArrayBuilder(unsigned int len, const std::shared_ptr<Element>& elem)
            : length(len), element(elem) {}

    std::shared_ptr<Element> build() const override {
        auto ret = std::make_shared<Element>(Type::ARRAY);
        ret->length = length;
        ret->list.push_back(element);
        ret->base_align = element->type == Type::STRUCT
                          ? element->base_align
                          : round_up_pow2(element->base_align, 4);
        return ret;
    }
};

class Element::ColMatBuilder : public Builder {
private:
    unsigned char cols;
    unsigned char rows;
public:
    ColMatBuilder(unsigned char c, unsigned char r) : cols(c), rows(r) {}

    std::shared_ptr<Element> build() const override {
        return ArrayBuilder(cols, VecBuilder(rows).build()).build();
    }
};

class Element::RowMatBuilder : public Builder {
private:
    unsigned char rows;
    unsigned char cols;
public:
    RowMatBuilder(unsigned char r, unsigned char c) : rows(r), cols(c) {}

    std::shared_ptr<Element> build() const override {
        return ArrayBuilder(rows, VecBuilder(cols).build()).build();
    }
};

class Element::StructBuilder : public Builder {
private:
    std::vector<std::shared_ptr<Element>> subelements;
public:
    explicit StructBuilder(const std::vector<std::shared_ptr<Element>>& subs)
            : subelements(subs) {}

    std::shared_ptr<Element> build() const override {
        auto ret = std::make_shared<Element>(Type::STRUCT);
        ret->list.insert(ret->list.end(), subelements.begin(), subelements.end());
        ret->length = static_cast<unsigned int>(ret->list.size());

        // base alignment is largest of its subelements
        if (!subelements.empty()) {
            for (const auto& e : subelements) {
                if (e->base_align > ret->base_align) {
                    ret->base_align = e->base_align;
                }
            }

            ret->base_align = round_up_pow2(ret->base_align, 4);
        }

        return ret;
    }
};



#endif //FEM_ENGINE_ELEMENT_H
